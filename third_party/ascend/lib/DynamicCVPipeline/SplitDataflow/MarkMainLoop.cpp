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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>

using namespace mlir;

static constexpr const char *DEBUG_TYPE = "mark-main-loop";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

// Which loop became the main loop and why. The answer is not derivable from the
// output IR: the attribute records the decision but not what drove it. Shares
// TRITON_ASCEND_CV_DEBUG_MAINLOOP with the reporting in AnalyzeScope, so one
// variable turns on the whole story.
static bool isMainLoopDebugEnabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("TRITON_ASCEND_CV_DEBUG_MAINLOOP");
    return env && *env && llvm::StringRef(env) != "0";
  }();
  return enabled;
}

using namespace mlir::triton;

namespace {

//===----------------------------------------------------------------------===//
// Selecting the main loop
//===----------------------------------------------------------------------===//
// The main loop is the loop across whose iterations the pipeline overlaps the
// two cores: everything downstream multi-buffers the handoff buffers, predicates
// the stages with scf.if and stretches the trip count to make room for a
// prologue and an epilogue. All of that is only meaningful for a loop whose body
// contains a *complete* producer -> consumer handoff. Half a handoff has nothing
// to rotate.
//
// The ledger for that is already in the IR when this pass runs, because
// InterCoreTransferAndSync has just written it: every cross-core exchange tags
// its two ends with `ssbuffer.crossCoreDeps = [group, role]`, role 1 = producer,
// role 0 = consumer, and the producer's `ssbuffer.core_type` gives the
// direction. It covers every channel -- the tensor transfers (hivm.hir.copy for
// V->C, hivm.hir.fixpipe for C->V), the scalar SSBuffer path (memref.store /
// memref.load), and the buffer-carried dependencies that move no data operation
// of their own and are tagged on the existing producer/consumer. C->C staging
// through L1 is deliberately absent from the ledger, which is what we want: it
// is cube-internal, not an exchange.
//
// The previous rule looked at operation kinds instead: it marked the nearest
// enclosing loop of any non-L1 hivm.hir.copy / hivm.hir.fixpipe and then kept
// only the innermost of any nested pair. Both halves misfire. It cannot see an
// exchange that has no data operation, and "innermost wins" strips the mark from
// the loop that actually carries the round trip whenever some nested loop
// happens to hold one leg of a different exchange -- which is exactly what a
// sparse-attention kernel does: the tile loop carries the full round trip, while
// a small inner loop wrapping the second matmul holds one C->V transfer, so the
// tile loop lost its mark to a two-iteration loop.

/// The chain of loops enclosing `op`, outermost first.
static SmallVector<Operation *> enclosingLoops(Operation *op) {
  SmallVector<Operation *> chain;
  for (Operation *cur = op->getParentOp(); cur; cur = cur->getParentOp()) {
    if (isa<scf::ForOp, scf::WhileOp>(cur)) {
      chain.push_back(cur);
    }
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

/// The innermost loop that encloses every op in `ops`, or null when they do not
/// share one. This is what makes a handoff "complete inside a loop".
static Operation *innermostCommonLoop(ArrayRef<Operation *> ops) {
  if (ops.empty()) {
    return nullptr;
  }
  SmallVector<Operation *> common = enclosingLoops(ops.front());
  for (Operation *op : ops.drop_front()) {
    SmallVector<Operation *> chain = enclosingLoops(op);
    size_t limit = std::min(common.size(), chain.size());
    size_t shared = 0;
    while (shared < limit && common[shared] == chain[shared]) {
      ++shared;
    }
    common.truncate(shared);
    if (common.empty()) {
      return nullptr;
    }
  }
  // Also covers the single-op case, where the loop above never runs.
  return common.empty() ? nullptr : common.back();
}

/// One cross-core exchange, gathered from `ssbuffer.crossCoreDeps`.
struct ExchangeGroup {
  SmallVector<Operation *> ops;
  bool hasProducer = false;
  bool hasConsumer = false;
  // Direction, taken from the producer's core type. V->C when the producer runs
  // on VECTOR, C->V otherwise.
  bool producerIsVector = false;
  bool directionKnown = false;

  bool isComplete() const { return hasProducer && hasConsumer; }
};

/// What a loop carries, counting every complete handoff nested anywhere inside
/// it, not only those whose innermost common loop is this one.
struct LoopCarriage {
  int handoffs = 0;
  bool v2c = false;
  bool c2v = false;

  bool isBidirectional() const { return v2c && c2v; }
};

/// Loops in deterministic program order, outermost of a nest first. DenseMap
/// iteration is not ordered, and main_loop ids must not depend on pointer
/// values; the outer-first order is what lets the selection below decide a nest
/// from the top down.
static SmallVector<Operation *> collectLoopsInOrder(ModuleOp module) {
  SmallVector<Operation *> loops;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<scf::ForOp, scf::WhileOp>(op)) {
      loops.push_back(op);
    }
  });
  return loops;
}

/// Read `ssbuffer.crossCoreDeps` into groups. Empty when the ledger is absent,
/// which happens for hand-written IR that never ran InterCoreTransferAndSync.
///
/// Direction comes from the producer's `ssbuffer.core_type`, which is still on
/// the operation here because this pass runs before SeparateCVScope. After the
/// split that attribute is gone (removeSsbufferAttrs) and the enclosing
/// scope.scope's `hivm.tcore_type` takes over, which is what AnalyzeScope reads.
/// The two are not interchangeable; each is the only record available where it
/// is used.
static llvm::DenseMap<int, ExchangeGroup> collectExchangeGroups(ModuleOp module) {
  llvm::DenseMap<int, ExchangeGroup> groups;
  module.walk([&](Operation *op) {
    auto depsAttr = op->getAttrOfType<ArrayAttr>(CVPipeline::kCrossCoreDeps);
    if (!depsAttr || depsAttr.size() < 2) {
      return;
    }
    auto groupAttr = dyn_cast<IntegerAttr>(depsAttr[0]);
    auto roleAttr = dyn_cast<IntegerAttr>(depsAttr[1]);
    if (!groupAttr || !roleAttr) {
      return;
    }
    ExchangeGroup &group = groups[static_cast<int>(groupAttr.getInt())];
    group.ops.push_back(op);
    if (roleAttr.getInt() == CVPipeline::crossCoreProducerId) {
      group.hasProducer = true;
      if (auto coreAttr =
              op->getAttrOfType<StringAttr>(CVPipeline::kCoreType)) {
        group.producerIsVector =
            coreAttr.getValue() == CVPipeline::kCoreTypeVector;
        group.directionKnown = true;
      }
    } else {
      group.hasConsumer = true;
    }
  });
  return groups;
}

/// Pick the main loops from the exchange ledger. Returns them in program order,
/// or empty when the ledger has nothing to say about any loop.
static SmallVector<Operation *> selectMainLoopsFromLedger(ModuleOp module) {
  llvm::DenseMap<int, ExchangeGroup> groups = collectExchangeGroups(module);
  if (groups.empty()) {
    return {};
  }

  // Attribute each complete handoff to its innermost enclosing loop and to
  // every loop above it: an outer loop carries what its inner loops carry.
  llvm::DenseMap<Operation *, LoopCarriage> carriage;
  SmallVector<int> groupIds;
  for (const auto &entry : groups) {
    groupIds.push_back(entry.first);
  }
  llvm::sort(groupIds);
  for (int groupId : groupIds) {
    const ExchangeGroup &group = groups[groupId];
    if (!group.isComplete()) {
      continue;
    }
    Operation *innermost = innermostCommonLoop(group.ops);
    if (!innermost) {
      // The two ends do not share a loop, so no loop can rotate this handoff.
      continue;
    }
    for (Operation *cur = innermost; cur; cur = cur->getParentOp()) {
      if (!isa<scf::ForOp, scf::WhileOp>(cur)) {
        continue;
      }
      LoopCarriage &facts = carriage[cur];
      ++facts.handoffs;
      if (group.directionKnown) {
        if (group.producerIsVector) {
          facts.v2c = true;
        } else {
          facts.c2v = true;
        }
      }
    }
  }

  SmallVector<Operation *> orderedLoops = collectLoopsInOrder(module);

  // Any loop that carries a complete handoff is a candidate, including a
  // one-way one: that is the epilogue-overlap shape, and it is for AnalyzeScope,
  // not for this pass, to decide whether to take it.
  SmallVector<Operation *> candidates;
  for (Operation *loopOp : orderedLoops) {
    auto it = carriage.find(loopOp);
    if (it != carriage.end() && it->second.handoffs > 0) {
      candidates.push_back(loopOp);
    }
  }

  auto isBidirectional = [&](Operation *loopOp) {
    auto it = carriage.find(loopOp);
    return it != carriage.end() && it->second.isBidirectional();
  };

  // Resolve each nest from the outside in. Within a nest the innermost
  // candidate normally wins -- it repeats most often and holds the least state
  // across an iteration -- but a loop that carries the round trip is not given
  // up for a nested loop that only carries one direction. That is the case this
  // rule exists for: the tile loop of a sparse-attention kernel holds the whole
  // CUBE<->VECTOR round trip, and a small inner loop around the second matmul
  // holds one C->V leg of it. Independent nests stay independent and each get
  // their own id; a nest never yields two ids, because everything downstream
  // pairs a main loop with exactly one partner in the other scope.
  SmallVector<Operation *> selected;
  for (Operation *loopOp : candidates) {
    bool ancestorAlreadyWon = false;
    for (Operation *winner : selected) {
      if (winner->isAncestor(loopOp)) {
        ancestorAlreadyWon = true;
        break;
      }
    }
    if (ancestorAlreadyWon) {
      continue;
    }

    bool beatenByNested = false;
    for (Operation *other : candidates) {
      if (other == loopOp || !loopOp->isAncestor(other)) {
        continue;
      }
      // A nested candidate takes over unless doing so would trade the round
      // trip for a single direction.
      if (!(isBidirectional(loopOp) && !isBidirectional(other))) {
        beatenByNested = true;
        break;
      }
    }
    if (!beatenByNested) {
      selected.push_back(loopOp);
    }
  }

  if (isMainLoopDebugEnabled()) {
    llvm::errs() << "[cv-mainloop] MarkMainLoop: exchange ledger has "
                 << groups.size() << " group(s)\n";
    for (Operation *loopOp : orderedLoops) {
      auto it = carriage.find(loopOp);
      if (it == carriage.end()) {
        continue;
      }
      llvm::errs() << "[cv-mainloop]   " << loopOp->getName().getStringRef()
                   << " carries " << it->second.handoffs << " handoff(s)"
                   << " V->C=" << (it->second.v2c ? "yes" : "no ")
                   << " C->V=" << (it->second.c2v ? "yes" : "no ") << " -> "
                   << (llvm::is_contained(selected, loopOp)
                           ? "SELECTED"
                           : "not selected")
                   << "\n";
    }
  }

  return selected;
}

/// The pre-existing heuristic: the nearest enclosing loop of any non-L1
/// hivm.hir.copy / hivm.hir.fixpipe, keeping only the innermost of a nested
/// pair. Kept as a last resort so that IR the ledger says nothing about -- a
/// hand-written test that never ran InterCoreTransferAndSync, say -- behaves
/// exactly as it did before.
static SmallVector<Operation *> selectMainLoopsLegacy(ModuleOp module) {
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

  SmallVector<Operation *> candidates;
  module.walk([&](Operation *op) {
    if (!isa<hivm::FixpipeOp, hivm::CopyOp>(op))
      return;

    if (isL1Fixpipe(op))
      return;

    if (auto forOp = op->getParentOfType<scf::ForOp>()) {
      if (!llvm::is_contained(candidates, forOp.getOperation()))
        candidates.push_back(forOp);
    }
    if (auto whileOp = op->getParentOfType<scf::WhileOp>()) {
      if (!llvm::is_contained(candidates, whileOp.getOperation()))
        candidates.push_back(whileOp);
    }
  });

  SmallVector<Operation *> selected;
  for (Operation *loopOp : candidates) {
    bool hasNestedCandidate = false;
    for (Operation *other : candidates) {
      if (other != loopOp && loopOp->isAncestor(other)) {
        hasNestedCandidate = true;
        break;
      }
    }
    if (!hasNestedCandidate) {
      selected.push_back(loopOp);
    }
  }

  if (isMainLoopDebugEnabled()) {
    llvm::errs() << "[cv-mainloop] MarkMainLoop: ledger named no loop, fell"
                    " back to the copy/fixpipe heuristic; "
                 << selected.size() << " of " << candidates.size()
                 << " candidate loop(s) selected\n";
  }

  return selected;
}

} // namespace

// Pass Entry Point
void MarkMainLoopPass::runOnOperation() {
  LOG_DEBUG("\n--- enter MarkMainLoopPass --->\n");
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  // The ledger decides when it has anything to say; the old heuristic only
  // covers IR that never ran InterCoreTransferAndSync. Selection already keeps
  // the innermost qualifying loop of a nest, so nothing has to be marked and
  // then unmarked -- ids no longer have gaps where a stripped loop used to sit.
  SmallVector<Operation *> selected = selectMainLoopsFromLedger(module);
  if (selected.empty()) {
    selected = selectMainLoopsLegacy(module);
  }

  int mainLoopIdCounter = 0;
  for (Operation *loopOp : selected) {
    if (loopOp->hasAttr(CVPipeline::kMainLoop)) {
      continue;
    }
    if (isMainLoopDebugEnabled()) {
      llvm::errs() << "[cv-mainloop] MarkMainLoop: "
                   << loopOp->getName().getStringRef()
                   << " -> ssbuffer.main_loop=" << mainLoopIdCounter << "\n";
    }
    // Add attribute with integer value (current counter ID)
    loopOp->setAttr(
        CVPipeline::kMainLoop,
        Builder(module.getContext()).getI32IntegerAttr(mainLoopIdCounter));
    mainLoopIdCounter++;
  }

  if (isMainLoopDebugEnabled()) {
    llvm::errs() << "[cv-mainloop] MarkMainLoop: " << mainLoopIdCounter
                 << " loop(s) marked\n";
  }

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
