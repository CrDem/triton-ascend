/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include <cstdlib>

static constexpr const char *DEBUG_TYPE = "split-gm-store-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

/// Off unless asked for. The pass trades a compute block for the chance to
/// overlap a vector iteration with the store of the previous one, and whether
/// that pays depends on the kernel, so it is opt-in.
static bool isEnabled() {
  static const bool enabled = []() -> bool {
    const char *env = std::getenv("TRITON_ASCEND_CV_SPLIT_GM_STORE_BLOCK");
    return env && llvm::StringRef(env) != "0";
  }();
  return enabled;
}

static bool isStoreOp(Operation *op) {
  return isa<bufferization::MaterializeInDestinationOp>(op) ||
         isa<hivm::StoreOp>(op);
}

static Value getStoreDest(Operation *storeOp) {
  if (auto materialize =
          dyn_cast<bufferization::MaterializeInDestinationOp>(storeOp)) {
    return materialize.getDest();
  }
  if (auto hivmStore = dyn_cast<hivm::StoreOp>(storeOp)) {
    return hivmStore.getDst();
  }
  return Value();
}

/// True when `v` is a view of a kernel argument, i.e. the store lands in GM.
static bool tracesToFuncArg(Value v) {
  while (true) {
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      break;
    }
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(defOp)) {
      v = viewLike.getViewSource();
      continue;
    }
    if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
      v = extractSlice.getSource();
      continue;
    }
    return false;
  }
  auto blockArg = dyn_cast<BlockArgument>(v);
  return blockArg && isa<func::FuncOp>(blockArg.getOwner()->getParentOp());
}

/// The store plus the view / extract_slice ops that feed it and are used by
/// nothing else. Pulling those along matters: left behind, the value crossing
/// into the new block would be the extract_slice result, whose shape is
/// dynamic, and AddMultiBufferInnerScope cannot allocate a buffer for that.
/// Moving them makes the crossing value the statically shaped tensor the
/// slice was taken from.
static llvm::SetVector<Operation *> collectStoreGroup(Operation *storeOp) {
  llvm::SetVector<Operation *> group;
  group.insert(storeOp);

  llvm::SmallVector<Operation *> worklist{storeOp};
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || group.contains(defOp)) {
        continue;
      }
      if (!isa<ViewLikeOpInterface>(defOp) &&
          !isa<tensor::ExtractSliceOp>(defOp)) {
        continue;
      }
      if (defOp->getBlock() != storeOp->getBlock()) {
        continue;
      }
      // Only ops the rest of the block no longer needs.
      bool usedElsewhere = llvm::any_of(defOp->getUsers(), [&](Operation *u) {
        return !group.contains(u);
      });
      if (usedElsewhere) {
        continue;
      }
      group.insert(defOp);
      worklist.push_back(defOp);
    }
  }
  return group;
}

class SplitGMStoreBlockPass
    : public PassWrapper<SplitGMStoreBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitGMStoreBlockPass)

  SplitGMStoreBlockPass() = default;

  StringRef getArgument() const override { return "split-gm-store-block"; }

  StringRef getDescription() const override {
    return "Give a GM store inside a loop its own vector compute block, so the "
           "stored value becomes a cross-block dependency that can be "
           "multi-buffered (opt-in via TRITON_ASCEND_CV_SPLIT_GM_STORE_BLOCK)";
  }

  void runOnOperation() override;
};

void SplitGMStoreBlockPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }
  if (!isEnabled()) {
    return;
  }

  auto bm = CVPipeline::ComputeBlockIdManager(module);

  llvm::SmallVector<Operation *> stores;
  module.walk([&](Operation *op) {
    if (!isStoreOp(op)) {
      return;
    }
    // Outside a loop the split buys nothing and costs a block.
    if (!op->getParentOfType<scf::ForOp>() &&
        !op->getParentOfType<scf::WhileOp>()) {
      return;
    }
    if (CVPipeline::getOpCoreType(op) != CVPipeline::CoreType::VECTOR_ONLY) {
      return;
    }
    Value dest = getStoreDest(op);
    if (!dest || !tracesToFuncArg(dest)) {
      return;
    }
    stores.push_back(op);
  });

  for (Operation *storeOp : stores) {
    auto group = collectStoreGroup(storeOp);

    // Nothing to separate when the block already holds only this group.
    auto blockId = bm.getBlockIdByOpOpt(storeOp);
    if (blockId.has_value()) {
      auto siblings = bm.getOpsByBlockId(*blockId);
      bool onlyGroup = llvm::all_of(siblings, [&](Operation *op) {
        return group.contains(op);
      });
      if (onlyGroup) {
        LOG_DEBUG("store already alone in block " << *blockId << ", skip");
        continue;
      }
    }

    llvm::SmallVector<Operation *> ordered(group.begin(), group.end());
    llvm::sort(ordered, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    // markOpsWithNewId refuses an op that already carries an id, and every op
    // here has one, so take a fresh id and force it in. updateBlockId rewrites
    // the attribute and moves the op between the manager's buckets; core_type
    // is left alone, which is what we want -- the ops stay on the vector core.
    const int newId = bm.getNextId();
    for (Operation *op : ordered) {
      bm.updateBlockId(op, newId);
    }
    LOG_DEBUG("moved a store group of " << ordered.size() << " op(s) into block "
                                        << newId);
  }
}

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createSplitGMStoreBlockPass() {
  return std::make_unique<SplitGMStoreBlockPass>();
}

} // namespace triton
} // namespace mlir
