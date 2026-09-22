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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/MarkMainLoop.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

static constexpr const char *DEBUG_TYPE = "mark-main-loop";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::triton;

// Pass Entry Point
//
// A "main loop" is the loop the dynamic CV pipeline is built around: its body is
// split into cube/vector stages that are then overlapped across iterations.
//
// Selection:
//   1. Candidates are the loops that (transitively) contain inter-core traffic,
//      i.e. a non-L1 hivm.copy / hivm.fixpipe.
//   2. `tl.range(..., main_loop=False)` removes a loop from the candidates.
//   3. By default the innermost candidate of a nest wins (historical behaviour).
//      `tl.range(..., main_loop=True)` overrides that: the hinted loop wins and
//      candidates nested inside it are dropped, unless they are hinted as well.
//      This matters for kernels whose innermost loop runs only a couple of
//      iterations - there the pipeline degenerates into prologue + epilogue,
//      while an outer loop would give the stages something to overlap with.
void MarkMainLoopPass::runOnOperation() {
  LOG_DEBUG("\n--- enter MarkMainLoopPass --->\n");
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  auto isL1Fixpipe = [](Operation *op) -> bool {
    auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(op);
    if (!fixpipeOp)
      return false;
    auto dstType = dyn_cast<MemRefType>(fixpipeOp.getDst().getType());
    if (!dstType)
      return false;
    auto addrSpaceAttr =
        dyn_cast_or_null<hivm::AddressSpaceAttr>(dstType.getMemorySpace());
    return addrSpaceAttr &&
           addrSpaceAttr.getAddressSpace() == hivm::AddressSpace::L1;
  };

  // Step 1: candidate loops - every loop that encloses inter-core traffic.
  // All enclosing loops are collected (not just the nearest one) so that an
  // explicit hint on an outer loop can be honoured.
  llvm::SetVector<Operation *> candidates;
  module.walk([&](Operation *op) {
    if (!isa<hivm::FixpipeOp, hivm::CopyOp>(op))
      return;
    if (isL1Fixpipe(op))
      return;
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (isa<scf::ForOp, scf::WhileOp>(parent))
        candidates.insert(parent);
    }
  });

  // Step 2: honour `main_loop=False`.
  llvm::SmallVector<Operation *> kept;
  for (Operation *loopOp : candidates) {
    if (CVPipeline::isMainLoopOptOut(loopOp)) {
      LOG_DEBUG("candidate dropped by main_loop=False hint\n");
      continue;
    }
    kept.push_back(loopOp);
  }

  // Step 3: resolve nests - an opted-in ancestor wins over inner candidates,
  // otherwise the innermost candidate wins.
  auto hasKeptDescendant = [&](Operation *loopOp) {
    for (Operation *other : kept) {
      if (other != loopOp && loopOp->isProperAncestor(other))
        return true;
    }
    return false;
  };
  auto hasOptedInAncestor = [&](Operation *loopOp) {
    for (Operation *other : kept) {
      if (other != loopOp && other->isProperAncestor(loopOp) &&
          CVPipeline::isMainLoopOptIn(other))
        return true;
    }
    return false;
  };

  llvm::SmallVector<Operation *> selected;
  for (Operation *loopOp : kept) {
    bool optedIn = CVPipeline::isMainLoopOptIn(loopOp);
    if (!optedIn && hasOptedInAncestor(loopOp)) {
      LOG_DEBUG("candidate dropped: enclosing loop is main_loop=True\n");
      continue;
    }
    if (!optedIn && hasKeptDescendant(loopOp)) {
      // Historical rule: keep only the innermost candidate.
      continue;
    }
    selected.push_back(loopOp);
  }

  // Step 4: tag the winners with dense ids in deterministic walk order.
  llvm::SmallPtrSet<Operation *, 8> selectedSet(selected.begin(),
                                                selected.end());
  int mainLoopIdCounter = 0;
  module.walk([&](Operation *loopOp) {
    if (!selectedSet.contains(loopOp))
      return;
    if (loopOp->hasAttr(CVPipeline::kMainLoop))
      return;
    loopOp->setAttr(
        CVPipeline::kMainLoop,
        Builder(module.getContext()).getI32IntegerAttr(mainLoopIdCounter));
    mainLoopIdCounter++;
  });

  LOG_DEBUG("--- exit MarkMainLoopPass --->\n");
}

// Create the pass
namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createMarkMainLoopPass() {
  return std::make_unique<MarkMainLoopPass>();
}
} // namespace triton
} // namespace mlir
