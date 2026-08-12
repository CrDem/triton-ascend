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

#include <memory>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/MainLoopUnroll.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/AddBlockIdForControlOps.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/InterCoreTransferAndSync.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/MarkMainLoop.h"

static constexpr const char *DEBUG_TYPE = "main-loop-unroll";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

namespace {

// Temporary tag put on every scf.for before probing, so that a main loop found
// in the probe clone can be matched back to the loop of the real module. It is
// always removed before the pass returns.
constexpr llvm::StringLiteral kMainLoopProbe = "ssbuffer.main_loop_probe";

class MainLoopUnrollPass
    : public ::impl::MainLoopUnrollBase<MainLoopUnrollPass> {
public:
  explicit MainLoopUnrollPass(const MainLoopUnrollOptions &options)
      : MainLoopUnrollBase(options) {}

  void runOnOperation() override;

private:
  // Run, on a throw-away clone, the passes that materialize the cube <-> vector
  // communication and mark the main loop. Returns the probe tags of the loops
  // `mark-main-loop` marked there.
  FailureOr<llvm::DenseSet<int>> probeMainLoops(ModuleOp module);

  // Shift the compute block ids of `root` and everything nested in it, so that
  // an unrolled copy gets compute blocks of its own.
  void shiftBlockIds(Operation *root, int copyIdx, int stride);
};

FailureOr<llvm::DenseSet<int>> MainLoopUnrollPass::probeMainLoops(
    ModuleOp module) {
  ModuleOp probe(module->clone());
  auto destroyProbe = llvm::make_scope_exit([&]() { probe->destroy(); });

  // These are the very passes SplitDataflow runs: the main loop is the loop
  // that ends up carrying the inter core transfers, so it can only be found
  // once those transfers have been inserted.
  PassManager pm(module.getContext(), module.getOperationName());
  pm.addPass(createAddBlockIdForControlOpsPass());
  pm.addPass(createDataDependencyAnalysisPass());
  pm.addPass(createInterCoreTransferAndSyncPass());
  pm.addPass(createMarkMainLoopPass());

  // Diagnostics of the probe run point at a module that is about to be thrown
  // away, so they would only confuse; a failure is reported by the caller.
  bool probeFailed = false;
  {
    ScopedDiagnosticHandler handler(module.getContext(),
                                    [](Diagnostic &) { return success(); });
    probeFailed = failed(pm.run(probe)) || CVPipeline::hasFallbackAttr(probe);
  }
  if (probeFailed) {
    return failure();
  }

  llvm::DenseSet<int> mainLoopTags;
  probe.walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr(CVPipeline::kMainLoop)) {
      return;
    }
    if (auto tag = forOp->getAttrOfType<IntegerAttr>(kMainLoopProbe)) {
      mainLoopTags.insert(static_cast<int>(tag.getInt()));
    }
  });
  return mainLoopTags;
}

void MainLoopUnrollPass::shiftBlockIds(Operation *root, int copyIdx,
                                       int stride) {
  if (copyIdx == 0) {
    return;
  }
  Builder builder(root->getContext());
  root->walk([&](Operation *op) {
    if (auto blockId = CVPipeline::getOpBlockId(op)) {
      op->setAttr(CVPipeline::kBlockId,
                  builder.getI32IntegerAttr(*blockId + copyIdx * stride));
    }
  });
}

void MainLoopUnrollPass::runOnOperation() {
  ModuleOp module = getOperation();
  const int factor = this->unrollFactor;

  if (factor <= 1) {
    LDBG("Unroll factor <= 1, nothing to do");
    return;
  }

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  Builder builder(module.getContext());
  int probeTag = 0;
  module.walk([&](scf::ForOp forOp) {
    forOp->setAttr(kMainLoopProbe, builder.getI32IntegerAttr(probeTag++));
  });
  if (probeTag == 0) {
    LDBG("Module has no loop, nothing to unroll");
    return;
  }
  auto removeTags = llvm::make_scope_exit([&]() {
    module.walk([](scf::ForOp forOp) { forOp->removeAttr(kMainLoopProbe); });
  });

  auto mainLoopTags = probeMainLoops(module);
  if (failed(mainLoopTags)) {
    module->emitWarning()
        << "[" << DEBUG_TYPE << "] "
        << "Could not determine the main loop, skipping the unroll by "
        << factor << ".";
    return;
  }
  if (mainLoopTags->empty()) {
    LDBG("No main loop found, nothing to unroll");
    return;
  }

  SmallVector<scf::ForOp> mainLoops;
  module.walk([&](scf::ForOp forOp) {
    auto tag = forOp->getAttrOfType<IntegerAttr>(kMainLoopProbe);
    if (tag && mainLoopTags->contains(static_cast<int>(tag.getInt()))) {
      mainLoops.push_back(forOp);
    }
  });

  // Each unrolled copy has to form compute blocks of its own: passes such as
  // inter-core-transfer-and-sync look a block up by id and would otherwise see
  // a single block spanning all the copies. Ids of the copies stay above the
  // ids already in use, so they keep growing along the program order.
  const int blockIdStride = CVPipeline::getAvailableBlockId(module);

  for (scf::ForOp forOp : mainLoops) {
    LDBG("Unrolling main loop by " << factor);
    auto unrolled = mlir::loopUnrollByFactor(
        forOp, static_cast<uint64_t>(factor),
        [&](unsigned copyIdx, Operation *clonedOp, OpBuilder) {
          shiftBlockIds(clonedOp, static_cast<int>(copyIdx), blockIdStride);
        });

    if (failed(unrolled)) {
      // Unrolling is an optimization: keep the original loop and go on.
      forOp->emitWarning() << "[" << DEBUG_TYPE << "] "
                           << "Failed to unroll the main loop by " << factor
                           << ", keeping it as is.";
      continue;
    }

    // The epilogue loop is a plain clone of the original loop, so its blocks
    // need to be renumbered as well.
    if (unrolled->epilogueLoopOp) {
      shiftBlockIds((*unrolled->epilogueLoopOp).getOperation(), factor,
                    blockIdStride);
    }
  }
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createMainLoopUnrollPass(const MainLoopUnrollOptions &options) {
  return std::make_unique<MainLoopUnrollPass>(options);
}
