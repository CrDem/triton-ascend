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

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <utility>

static constexpr const char *DEBUG_TYPE = "analyze-scope";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

using namespace llvm;
using namespace mlir;
using namespace triton;
using namespace CVPipeline;

namespace {
static bool isVectorScope(scope::ScopeOp scopeOp) {
  auto coreTypeAttr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  if (!coreTypeAttr) {
    return false;
  }
  return coreTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR;
}

// Which core an operation belongs to, by the scope it sits in.
//
// SeparateCVScope strips `ssbuffer.core_type` from every operation in the
// function once the two scopes exist (removeSsbufferAttrs), so by the time this
// pass runs the enclosing scope.scope's `hivm.tcore_type` is the only surviving
// record of it. MarkMainLoop, which runs *before* the split, reads
// `ssbuffer.core_type` for the same purpose -- the two are not interchangeable,
// each is the only thing available where it is used.
static std::optional<bool> runsOnVectorCore(Operation *op) {
  auto scopeOp = op->getParentOfType<scope::ScopeOp>();
  if (!scopeOp) {
    return std::nullopt;
  }
  auto coreTypeAttr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  if (!coreTypeAttr) {
    return std::nullopt;
  }
  return coreTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR;
}

static bool checkTransferInteraction(mlir::Operation *op) {
  bool hasCVInteraction = false;
  // check v->c data interaction
  if (isa<hivm::CopyOp>(op)) {
    hasCVInteraction = true;
  }
  // check c->v data interaction
  // user with "ssbuffer.add_from_matmul" is not a real c->v data interaction
  if (isa<bufferization::ToTensorOp>(op)) {
    for (Operation *user : op->getUsers()) {
      if (!user->hasAttr(CVPipeline::kAddFromMatmul)) {
        hasCVInteraction = true;
        break;
      }
      for (Operation *userUser : user->getUsers()) {
        if (!isa<scf::YieldOp>(userUser)) {
          hasCVInteraction = true;
          break;
        }
      }
    }
  }

  return hasCVInteraction;
}

static bool checkVecScopeMainLoop(ModuleOp module) {
  bool hasMainLoop = false;
  bool allMainLoopsSatisfy = true;

  module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    if (!isVectorScope(scopeOp)) {
      return WalkResult::advance();
    }

    scopeOp.walk([&](Operation *op) -> WalkResult {
      if (!isMainLoopOp(op)) {
        return WalkResult::advance();
      }

      hasMainLoop = true;
      bool hasCVInteraction = false;

      op->walk([&](mlir::Operation *innerOp) -> WalkResult {
        if (innerOp == op) {
          return WalkResult::advance();
        }

        // ops with "ssbuffer.transfer_id" are injected in SplitDataflowPass for
        // data transfer
        if (innerOp->hasAttr(CVPipeline::kTransferId)) {
          hasCVInteraction = checkTransferInteraction(innerOp);
          if (hasCVInteraction) {
            return WalkResult::interrupt();
          }
        }

        return WalkResult::advance();
      });

      // As long as there is a for/while with "ssbuffer.main_loop" that is not a
      // real mainloop, the processing conditions are not met, need to skip.
      if (!hasCVInteraction) {
        allMainLoopsSatisfy = false;
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    if (!allMainLoopsSatisfy) {
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return hasMainLoop && allMainLoopsSatisfy;
}

// The pre-existing form of the one-way check: for every main_loop id, count the
// hivm.hir.copy and hivm.hir.fixpipe ops inside the loops carrying that id, and
// declare the pipeline inapplicable when no id has both.
//   - hivm::CopyOp    typically appears in VECTOR scope main_loops
//   - hivm::FixpipeOp typically appears in CUBE scope main_loops
//
// Counting operation kinds is only a proxy for "does this loop exchange data in
// both directions", and it is wrong in three ways, so noMainLoopCanBePipelined
// below reads the exchange ledger instead and this is kept only for IR that has
// no ledger at all. The three:
//   - a V->C exchange carried by a *buffer* has no data operation of its own
//     (handleMemoryDependency only emits sync_block_set/wait on PIPE_MTE2 and
//     tags the existing producer/consumer with memCrossDeps), and the CUBE side
//     reads it with memref.copy, which is not hivm::CopyOp;
//   - a scalar V->C exchange travels through SSBuffer as memref.store /
//     memref.load on PIPE_S, also not hivm::CopyOp;
//   - a C->C fixpipe staging through L1 is counted as though it were C->V, even
//     though MarkMainLoop deliberately excludes it (isL1Fixpipe).
//
// Nested regions inside the main_loop op are also walked, and scf.yield
// terminators are skipped.
static bool isEveryMainLoopOneWayByOpKind(ModuleOp module) {
  // main_loop id -> (countCopy, countFixpipe)
  llvm::DenseMap<int, std::pair<int, int>> idToCounts;

  module.walk([&](Operation *op) -> WalkResult {
    if (!isa<scf::ForOp, scf::WhileOp>(op)) {
      return WalkResult::advance();
    }
    auto mainLoopAttr = op->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop);
    if (!mainLoopAttr) {
      return WalkResult::advance();
    }

    int id = mainLoopAttr.getInt();
    auto &counts = idToCounts[id];

    op->walk([&](mlir::Operation *innerOp) -> WalkResult {
      // Skip the loop op itself (the walk visits it first)
      if (innerOp == op) {
        return WalkResult::advance();
      }
      // Skip yield terminators (they are not real ops)
      if (isa<scf::YieldOp>(innerOp)) {
        return WalkResult::advance();
      }
      if (isa<hivm::CopyOp>(innerOp)) {
        ++counts.first;
      } else if (isa<hivm::FixpipeOp>(innerOp)) {
        ++counts.second;
      }
      return WalkResult::advance();
    });

    return WalkResult::advance();
  });

  // Only trigger fallback when EVERY main_loop id has only copy or only
  // fixpipe (one count is zero). If at least one id has both copy and
  // fixpipe ops, the dynamic CV pipeline still be applicable.
  if (idToCounts.empty()) {
    return false;
  }

  for (const auto &entry : idToCounts) {
    if (entry.second.first != 0 && entry.second.second != 0) {
      return false;
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// The exchange ledger
//===----------------------------------------------------------------------===//
// InterCoreTransferAndSync tags both ends of every cross-core exchange with
// `ssbuffer.crossCoreDeps = [group, role]`, role 1 = producer, role 0 =
// consumer; the direction is the core the producer runs on, which here means
// the scope it sits in (see runsOnVectorCore). It covers every channel: tensor
// transfers, the scalar SSBuffer path, and buffer-carried dependencies. C->C
// staging through L1 is deliberately not in it, which is right -- it is
// cube-internal, not an exchange.
//
// This is the same data the rest of the pipeline runs on: AddControlFlowCondition
// builds its whole stage predication from it (InitDependentMap::collectDepsByGroup),
// direction-agnostically. Deciding applicability from anything else means
// deciding it from a different picture of the kernel than the one that will be
// compiled.
//
// The two ends of a group live in different scopes after SeparateCVScope -- the
// producer in one scope's clone of the main loop, the consumer in the other's --
// and both clones carry the same main_loop id, so gathering per id sees both.

/// The roles of one exchange group seen inside the loops of one main_loop id.
struct GroupRoles {
  bool hasProducer = false;
  bool hasConsumer = false;
  bool producerIsVector = false;
  bool directionKnown = false;
  // Which machinery carries this exchange, which is what decides whether it can
  // survive having its two ends put in different pipeline stages:
  //
  //   - a *transfer* group (ssbuffer.transfer_id) gets the full handshake from
  //     insertInterCoreSync -- "data ready" forward and "buffer free" back, plus
  //     a credit before the loop and a drain after it -- and a rotatable buffer
  //     from AllocMultiCache. Overlapping iterations is safe.
  //   - a *buffer-carried* group (ssbuffer.memCrossDeps) gets only the forward
  //     half from insertMemDepSync: one set, one wait, no back-signal, and the
  //     buffer is the kernel's own memory, so nothing rotates it. Correct while
  //     the two ends stay in lockstep, and a write-after-read race the moment
  //     they do not.
  bool viaTransfer = false;
  bool viaMemDep = false;

  bool isComplete() const { return hasProducer && hasConsumer; }
};

/// main_loop id -> exchange group -> roles seen inside that id's loops.
using LedgerByMainLoop = llvm::DenseMap<int, llvm::DenseMap<int, GroupRoles>>;

static LedgerByMainLoop collectLedgerByMainLoop(ModuleOp module) {
  LedgerByMainLoop ledger;

  module.walk([&](Operation *loopOp) {
    if (!isa<scf::ForOp, scf::WhileOp>(loopOp)) {
      return;
    }
    auto mainLoopAttr =
        loopOp->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop);
    if (!mainLoopAttr) {
      return;
    }
    const int id = static_cast<int>(mainLoopAttr.getInt());
    // Make sure the id is present even when it carries no exchange at all: an
    // id with an empty group map is a main loop with nothing to overlap, and
    // the caller has to be able to tell that from "no main loops".
    llvm::DenseMap<int, GroupRoles> &groups = ledger[id];

    loopOp->walk([&](Operation *op) {
      auto depsAttr = op->getAttrOfType<ArrayAttr>(CVPipeline::kCrossCoreDeps);
      if (!depsAttr || depsAttr.size() < 2) {
        return;
      }
      auto groupAttr = dyn_cast<IntegerAttr>(depsAttr[0]);
      auto roleAttr = dyn_cast<IntegerAttr>(depsAttr[1]);
      if (!groupAttr || !roleAttr) {
        return;
      }
      GroupRoles &roles = groups[static_cast<int>(groupAttr.getInt())];
      if (op->hasAttr(CVPipeline::kTransferId)) {
        roles.viaTransfer = true;
      }
      if (op->hasAttr(CVPipeline::kMemCrossDeps)) {
        roles.viaMemDep = true;
      }
      if (roleAttr.getInt() == CVPipeline::crossCoreProducerId) {
        roles.hasProducer = true;
        if (std::optional<bool> onVector = runsOnVectorCore(op)) {
          roles.producerIsVector = *onVector;
          roles.directionKnown = true;
        }
      } else {
        roles.hasConsumer = true;
      }
    });
  });

  return ledger;
}

/// What one main_loop id offers the pipeline, and what it costs.
struct MainLoopVerdict {
  /// At least one complete handoff carried by real transfer machinery, so there
  /// is something to overlap across iterations.
  bool worthwhile = false;
  /// No buffer-carried dependency has both of its ends inside this loop.
  bool safe = true;

  bool applicable() const { return worthwhile && safe; }
};

static MainLoopVerdict
judgeMainLoop(const llvm::DenseMap<int, GroupRoles> &groups) {
  MainLoopVerdict verdict;
  for (const auto &entry : groups) {
    const GroupRoles &roles = entry.second;
    // Half a handoff has nothing to rotate across iterations, and it is also
    // not a hazard: the other end is outside, in lockstep with the loop.
    if (!roles.isComplete()) {
      continue;
    }
    if (roles.viaTransfer) {
      verdict.worthwhile = true;
    }
    if (roles.viaMemDep && !roles.viaTransfer) {
      verdict.safe = false;
    }
  }
  return verdict;
}

/// True when no main_loop id can be pipelined, which is when the dynamic CV
/// pipeline gives up and the original workflow is used.
///
/// This replaces the older "a main loop must carry a round trip" rule, which
/// asked the wrong question in both directions.
///
/// It was too strict about *direction*. A one-way CUBE->VECTOR loop -- cube
/// computes, vector post-processes and writes out -- is the plain epilogue
/// overlap, and the machinery that makes it safe is per transfer and blind to
/// direction: AllocMultiCache rotates the buffer, insertInterCoreSync guards it
/// with a handshake in both directions regardless of which way the data flows.
/// Nothing about a second direction is load-bearing. Requiring one rejected
/// every kernel whose vector side is a pure epilogue.
///
/// It was too lax about *machinery*. Counting operation kinds happened to count
/// exactly the exchanges that have the full handshake, so the old rule was
/// accidentally safe; reading the ledger instead admits buffer-carried
/// dependencies, and those have only a forward signal and an unrotated buffer.
/// Put their two ends in different pipeline stages and the producer of
/// iteration i+1 overwrites what the consumer of iteration i is still reading --
/// a write-after-read race with no flag to catch it, deterministic, and
/// invisible to a flag-pairing check because both sides are perfectly balanced.
///
/// So the two questions are asked separately: is there a transfer-carried
/// handoff to overlap, and is there a buffer-carried one that would be torn
/// apart. Once insertMemDepSync emits the back-signal that insertInterCoreSync
/// already does, the safety half of this can go away.
static bool noMainLoopCanBePipelined(ModuleOp module) {
  LedgerByMainLoop ledger = collectLedgerByMainLoop(module);

  // No main loop at all: preserve the previous answer, which left the decision
  // to the earlier gates.
  if (ledger.empty()) {
    return false;
  }

  bool sawAnyGroup = false;
  for (const auto &entry : ledger) {
    if (!entry.second.empty()) {
      sawAnyGroup = true;
      break;
    }
  }
  // No ledger to read -- hand-written IR that never ran
  // InterCoreTransferAndSync. Answer exactly as before.
  if (!sawAnyGroup) {
    return isEveryMainLoopOneWayByOpKind(module);
  }

  for (const auto &entry : ledger) {
    if (judgeMainLoop(entry.second).applicable()) {
      return false;
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Applicability diagnostics
//===----------------------------------------------------------------------===//
// The three gates in verifyMainLoop decide whether the dynamic CV pipeline
// applies, and two of them decide it by counting operation kinds instead of
// reading the exchange ledger the rest of the pipeline actually runs on:
// `ssbuffer.transfer_id` plus `ssbuffer.crossCoreDeps = [group, role]`, role 1
// = producer, 0 = consumer (see InitDependentMap::collectDepsByGroup).
//
// The two views disagree in three known ways, all of which make a rejection
// impossible to judge from the counters alone:
//
//   - a *scalar* V->C exchange travels through SSBuffer as memref.store /
//     memref.load on PIPE_S, not as hivm.hir.copy, so the counters miss it
//     entirely and a loop with tensor C->V plus scalar V->C reads as one-way;
//   - a *buffer-carried* cross-core dependency moves no data operation at all
//     (handleMemoryDependency only emits sync_block_set/wait on PIPE_MTE2 and
//     tags the existing producer/consumer with memCrossDeps), so the counters
//     miss it while the ledger has it;
//   - a C->C fixpipe staging through L1 is counted as though it were C->V,
//     even though MarkMainLoop deliberately excludes it (isL1Fixpipe).
//
// This prints both views side by side so the disagreement is visible. Off
// unless TRITON_ASCEND_CV_DEBUG_MAINLOOP is set to something other than 0.
//
// Note: the variant search in AddDynamicCVPipeline silences stdout/stderr
// while it evaluates candidates, so run without TRITON_ASCEND_CV_VARIANTS (or
// read the final, unsilenced compilation) to see this.
static bool isMainLoopDebugEnabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("TRITON_ASCEND_CV_DEBUG_MAINLOOP");
    return env && *env && llvm::StringRef(env) != "0";
  }();
  return enabled;
}

enum class FixpipeDst { UB, L1, Unknown };

static FixpipeDst classifyFixpipeDst(hivm::FixpipeOp fixpipeOp) {
  auto dstType = dyn_cast<MemRefType>(fixpipeOp.getDst().getType());
  if (!dstType) {
    return FixpipeDst::Unknown;
  }
  auto addrSpaceAttr =
      dyn_cast_or_null<hivm::AddressSpaceAttr>(dstType.getMemorySpace());
  if (!addrSpaceAttr) {
    return FixpipeDst::Unknown;
  }
  switch (addrSpaceAttr.getAddressSpace()) {
  case hivm::AddressSpace::UB:
    return FixpipeDst::UB;
  case hivm::AddressSpace::L1:
    return FixpipeDst::L1;
  default:
    return FixpipeDst::Unknown;
  }
}

struct MainLoopFacts {
  // How many loop ops carry this id. Normally two after SeparateCVScope: the
  // VECTOR clone and the CUBE clone.
  int loops = 0;
  // What the old op-kind check counted.
  int copies = 0;
  int fixpipes = 0;
  // The same fixpipes, split by destination.
  int fixpipeToUB = 0;
  int fixpipeToL1 = 0;
  int fixpipeUnknown = 0;
  // The scalar SSBuffer channel, which the counters above cannot see.
  int scalarStores = 0;
  int scalarLoads = 0;
  int transferTaggedOps = 0;
  // Ops carrying ssbuffer.memCrossDeps: a cross-core dependency carried by a
  // buffer rather than by a value. It moves no data of its own, so it is
  // invisible to the copy/fixpipe counters, but it is a real direction of the
  // exchange and it is in the crossCoreDeps ledger.
  int memDepOps = 0;
};

static void reportMainLoopFacts(ModuleOp module) {
  if (!isMainLoopDebugEnabled()) {
    return;
  }

  llvm::DenseMap<int, MainLoopFacts> byId;
  // The same ledger the decision reads, so the report and the verdict can never
  // disagree about what is inside the loop.
  LedgerByMainLoop ledger = collectLedgerByMainLoop(module);

  module.walk([&](Operation *loopOp) {
    if (!isa<scf::ForOp, scf::WhileOp>(loopOp)) {
      return;
    }
    auto mainLoopAttr =
        loopOp->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop);
    if (!mainLoopAttr) {
      return;
    }
    const int id = static_cast<int>(mainLoopAttr.getInt());
    MainLoopFacts &facts = byId[id];
    ++facts.loops;

    loopOp->walk([&](Operation *op) {
      if (op == loopOp || isa<scf::YieldOp>(op)) {
        return;
      }

      if (isa<hivm::CopyOp>(op)) {
        ++facts.copies;
      } else if (auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(op)) {
        ++facts.fixpipes;
        switch (classifyFixpipeDst(fixpipeOp)) {
        case FixpipeDst::UB:
          ++facts.fixpipeToUB;
          break;
        case FixpipeDst::L1:
          ++facts.fixpipeToL1;
          break;
        case FixpipeDst::Unknown:
          ++facts.fixpipeUnknown;
          break;
        }
      }

      if (op->hasAttr(CVPipeline::kTransferId)) {
        ++facts.transferTaggedOps;
        if (isa<memref::StoreOp>(op)) {
          ++facts.scalarStores;
        }
        if (isa<memref::LoadOp>(op)) {
          ++facts.scalarLoads;
        }
      }
      if (op->hasAttr(CVPipeline::kMemCrossDeps)) {
        ++facts.memDepOps;
      }
    });
  });

  llvm::errs() << "[cv-mainloop] ==== AnalyzeScope: main loop facts ====\n";

  if (byId.empty()) {
    // Gate 1 territory. Say what exists instead, so "no main loop" can be told
    // apart from "no cross-core exchange anywhere".
    int copies = 0, fixpipes = 0, tagged = 0, loops = 0;
    module.walk([&](Operation *op) {
      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        ++loops;
      }
      if (isa<hivm::CopyOp>(op)) {
        ++copies;
      }
      if (isa<hivm::FixpipeOp>(op)) {
        ++fixpipes;
      }
      if (op->hasAttr(CVPipeline::kTransferId)) {
        ++tagged;
      }
    });
    llvm::errs() << "[cv-mainloop] no loop carries ssbuffer.main_loop -> gate 1"
                    " (hasMainLoopOp) will reject\n"
                 << "[cv-mainloop]   module-wide: loops=" << loops
                 << " copy=" << copies << " fixpipe=" << fixpipes
                 << " transfer-tagged ops=" << tagged << "\n"
                 << "[cv-mainloop]   (transfers present but no marked loop"
                    " means the exchange landed outside every loop)\n";
    return;
  }

  SmallVector<int> ids;
  for (const auto &entry : byId) {
    ids.push_back(entry.first);
  }
  llvm::sort(ids);

  bool anyIdTwoWayByCounters = false;
  for (int id : ids) {
    const MainLoopFacts &facts = byId[id];
    const bool twoWayByCounters = facts.copies != 0 && facts.fixpipes != 0;
    anyIdTwoWayByCounters |= twoWayByCounters;

    llvm::errs() << "[cv-mainloop] main_loop id=" << id << " (" << facts.loops
                 << " loop op(s) carry this id)\n"
                 << "[cv-mainloop]   gate-3 counters : copy=" << facts.copies
                 << " fixpipe=" << facts.fixpipes << "  -> "
                 << (twoWayByCounters ? "two-way (accepts)"
                                      : "ONE-WAY (rejects)")
                 << "\n"
                 << "[cv-mainloop]   fixpipe by dst  : ub(C->V)="
                 << facts.fixpipeToUB
                 << " l1(C->C staging, miscounted as C->V)="
                 << facts.fixpipeToL1 << " unknown=" << facts.fixpipeUnknown
                 << "\n"
                 << "[cv-mainloop]   scalar channel  : memref.store="
                 << facts.scalarStores << " memref.load=" << facts.scalarLoads
                 << "  (V->C via SSBuffer; invisible to gate 3)\n"
                 << "[cv-mainloop]   memCrossDeps ops=" << facts.memDepOps
                 << "  (buffer-carried exchange; moves no data op, so also"
                    " invisible to gate 3)\n"
                 << "[cv-mainloop]   transfer-tagged ops="
                 << facts.transferTaggedOps << "\n";

    const llvm::DenseMap<int, GroupRoles> &idGroups = ledger[id];
    int complete = 0, incomplete = 0, v2c = 0, c2v = 0, unknownDir = 0;
    SmallVector<int> groups;
    for (const auto &entry : idGroups) {
      groups.push_back(entry.first);
    }
    llvm::sort(groups);
    for (int group : groups) {
      const GroupRoles roles = idGroups.lookup(group);
      if (roles.isComplete()) {
        ++complete;
      } else {
        ++incomplete;
      }
      if (!roles.directionKnown) {
        ++unknownDir;
      } else if (roles.producerIsVector) {
        ++v2c;
      } else {
        ++c2v;
      }
      llvm::errs() << "[cv-mainloop]     group " << group << ": producer="
                   << (roles.hasProducer ? "yes" : "no ") << " consumer="
                   << (roles.hasConsumer ? "yes" : "no ") << " direction="
                   << (!roles.directionKnown
                           ? llvm::StringRef("?")
                           : (roles.producerIsVector ? llvm::StringRef("V->C")
                                                     : llvm::StringRef("C->V")))
                   << " via="
                   << (roles.viaTransfer ? "transfer"
                                         : (roles.viaMemDep ? "memdep" : "?"))
                   << (roles.isComplete() ? "  [complete handoff]"
                                          : "  [INCOMPLETE]")
                   << "\n";
    }

    const MainLoopVerdict verdict = judgeMainLoop(idGroups);
    llvm::errs() << "[cv-mainloop]   ledger view     : " << complete
                 << " complete handoff(s), " << incomplete
                 << " incomplete; directions V->C=" << v2c << " C->V=" << c2v
                 << " unknown=" << unknownDir << "\n"
                 << "[cv-mainloop]   verdict         : "
                 << (verdict.worthwhile ? "worthwhile (a transfer handoff to"
                                          " overlap)"
                                        : "NOT worthwhile (no complete"
                                          " transfer handoff)")
                 << ", "
                 << (verdict.safe
                         ? "safe"
                         : "UNSAFE (a buffer-carried handoff would be split"
                           " across stages; insertMemDepSync emits no"
                           " back-signal, so that is a write-after-read race)")
                 << "\n";
    if (verdict.applicable() && !twoWayByCounters) {
      llvm::errs() << "[cv-mainloop]   note: the old op-kind rule would have"
                      " rejected this id as one-way. A one-way transfer handoff"
                      " is the plain epilogue overlap and is rotated by the"
                      " same machinery as a round trip.\n";
    }
  }

  const bool blocked = noMainLoopCanBePipelined(module);
  llvm::errs() << "[cv-mainloop] applicability: "
               << (blocked ? "no id is both worthwhile and safe -> FALLBACK"
                           : "at least one id can be pipelined -> proceed")
               << " (op-kind counters alone would have said "
               << (anyIdTwoWayByCounters ? "proceed" : "FALLBACK") << ")\n";
}

static LogicalResult verifyMainLoop(ModuleOp module) {
  reportMainLoopFacts(module);

  bool hasMainLoopOp = false;
  module.walk([&](Operation *op) {
    if (isMainLoopOp(op)) {
      hasMainLoopOp = true;
    }
  });

  if (!hasMainLoopOp) {
    LDBG("[INFO]: No cycle of multiple iterations, the DynamicCVPipeline pass "
         "will be interrupted, and resumed to the original workflow.");
    std::cout << "[DEBUG VDV] AnalyzeScope hasMainLoopForOp failed" << std::endl;
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return failure();
  }

  if (!checkVecScopeMainLoop(module)) {
    LDBG("[INFO]: No op beside matmul add in vector main loop.");
    std::cout << "[DEBUG VDV] AnalyzScope checkVecScopeMainLoop failed" << std::endl;
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return failure();
  };

  if (noMainLoopCanBePipelined(module)) {
    LDBG("[INFO]: No main loop carries a pipelineable CV exchange.");
    std::cout << "[VDV DEBUG] AnalyzeScope - no main_loop carries a transfer handoff that is safe to pipeline. - FALLBACK" << std::endl;
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return failure();
  }

  return success();
}

} // namespace

void AnalyzeScopePass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Before AnalyzeScope:\n" << module << "\n");

  if (failed(verifyMainLoop(module))) {
    std::cout << "[DEBUG VDV] AnalyzScope verifyMainLoop failed" << std::endl;
    return;
  }

  LDBG("After AnalyzeScope:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeScopePass() {
  return std::make_unique<AnalyzeScopePass>();
}

} // namespace triton
} // namespace mlir
