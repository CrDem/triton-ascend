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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache.h"
#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/EstimateCVPipelineCost.h"
#include "ascend/include/DynamicCVPipeline/MainLoopUnroll.h"
#include "ascend/include/DynamicCVPipeline/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlockPass.h"
#include "ascend/include/DynamicCVPipeline/PreCheckAvailable.h"
#include "ascend/include/DynamicCVPipeline/RemoveAttributes.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromComputePass.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflowPass.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <tuple>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

static constexpr const char *DEBUG_TYPE = "AddDynamicCVPipeline";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ADDDYNAMICCVPIPELINE
#include "ascend/include/DynamicCVPipeline/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Variant search
//===----------------------------------------------------------------------===//
// The pipeline is deterministic apart from one decision: the order the
// operations are linearised in, which ReorderOpsByBlockId makes and which
// ssbuffer.reorder_seed selects. Everything downstream -- where the blocks fall,
// where the barriers land, how deep the software pipeline gets -- follows from
// it, so running the pipeline once per seed and keeping the cheapest result is
// a search over genuinely different IR rather than over cosmetic orderings.
//
// The cost of the search is one full pipeline run per candidate, so it is off
// unless asked for. Candidates that fail are simply skipped: a variant that
// trips a downstream assumption sets the fallback attribute, and the search
// treats that as an infinitely expensive candidate rather than as an error.

constexpr const char *kVariantCountEnvVar = "TRITON_ASCEND_CV_VARIANTS";

/// Two totals this close to each other are treated as the same number.
///
/// The estimate is the larger of two bounds, and the resource bound does not
/// depend on how operations were grouped -- so once a candidate reaches it,
/// every other candidate that also reaches it scores identically, and the
/// search has nothing left to compare. Exact equality is the common case;
/// near-equality happens because the fusion factor rounds per block, and one
/// run had a candidate win by a single cycle out of half a billion. Neither is
/// a finding, so both are called a tie and decided on the second bound.
///
/// 0.0001 is deliberately tight. Widening it starts trading a real difference
/// in the leading bound for a difference in the trailing one, which is the
/// wrong way round.
constexpr int64_t kTieRelativeDenominator = 10000;

bool totalsAreTied(int64_t a, int64_t b) {
  const int64_t diff = a > b ? a - b : b - a;
  const int64_t scale = a > b ? a : b;
  return diff * kTieRelativeDenominator <= scale;
}

constexpr const char *kUBSlackEnvVar = "TRITON_ASCEND_CV_UB_SLACK_BYTES";

/// How much more Unified Buffer a candidate may hold live than the untouched
/// pipeline does, before the search discards it.
///
/// The search has no other way to know that a candidate cannot be built:
/// overrunning UB is diagnosed by the binary compiler, in a separate process,
/// after the whole of this compilation has finished. So an unbuildable
/// candidate looks perfectly healthy here and gets picked -- and it is the good
/// candidates that overrun, because the overlap they win is paid for by
/// materialising the values that cross each new block boundary. Estimate and
/// footprint are the same variable measured twice.
///
/// Measured against the baseline rather than against the capacity, and that is
/// the whole point. The absolute figure this model computes is not the one the
/// binary compiler enforces -- it multi-buffers again on top, aligns to banks,
/// adds temporaries later -- but every one of those applies to the baseline
/// too, so they cancel in a difference. What is left is the increment, which is
/// exactly the buffers the new cuts created, and that this model does know.
/// The baseline compiles by construction: it is what the pipeline emits with
/// none of this, and the search spends its first attempt on it.
///
/// Unset, it is derived rather than defaulted: the room a candidate may spend
/// is exactly what the baseline leaves unused, `capacity - peak(baseline)`,
/// both halves of which the estimate publishes. That is automatic on any
/// kernel -- a baseline that already fills the buffer yields zero and admits
/// nothing, which is the right answer and is what flash attention at 128x128
/// gives. Set, it overrides: 0 admits only candidates holding no more than the
/// baseline, 65536 lets one more score tile through.
constexpr const char *kStageSlackEnvVar = "TRITON_ASCEND_CV_STAGE_SLACK";

/// How many more iterations of software-pipeline prologue a candidate may
/// carry than the untouched pipeline does.
///
/// The third and last thing a finer block partition spends. More blocks means
/// more pipeline stages, and UpdateLoopIterTimes stretches the loop bound to
/// make room for them: the untouched pipeline stretches by 2 iterations, and
/// every candidate the binary compiler has refused for 'read before first
/// write' stretched by more than 130. That error *is* the prologue -- an
/// iteration in which a stage reads a buffer whose filling stage has not run.
///
/// Defaults to zero, which is not a tuned value but the structural rule: the
/// depth is a small integer the pipeline chose, and a candidate that needs a
/// deeper one has asked for a transformation the toolchain then refuses. Unlike
/// an extension in iterations, nothing here scales with the kernel, so there is
/// nothing to fit per kernel. The variable only widens it.
std::optional<int64_t> getStageSlack() {
  static const std::optional<int64_t> slack = []() -> std::optional<int64_t> {
    const char *env = std::getenv(kStageSlackEnvVar);
    if (!env) {
      return std::nullopt;
    }
    int64_t value = 0;
    if (llvm::StringRef(env).getAsInteger(10, value) || value < 0) {
      llvm::errs() << "[" << DEBUG_TYPE << "] " << kStageSlackEnvVar << "='"
                   << env << "' is not a count; ignoring\n";
      return std::nullopt;
    }
    return value;
  }();
  return slack;
}

std::optional<int64_t> getUBSlackBytes() {
  static const std::optional<int64_t> slack = []() -> std::optional<int64_t> {
    const char *env = std::getenv(kUBSlackEnvVar);
    if (!env) {
      return std::nullopt;
    }
    int64_t value = 0;
    if (llvm::StringRef(env).getAsInteger(10, value) || value < 0) {
      llvm::errs() << "[" << DEBUG_TYPE << "] " << kUBSlackEnvVar << "='" << env
                   << "' is not a byte count; ignoring\n";
      return std::nullopt;
    }
    return value;
  }();
  return slack;
}

/// How many orderings to try. One or fewer means the ordinary single
/// compilation, which is what every build that does not ask for a search gets.
int getVariantCount() {
  const char *env = std::getenv(kVariantCountEnvVar);
  if (!env) {
    return 1;
  }
  int value = 0;
  if (llvm::StringRef(env).getAsInteger(10, value) || value < 1) {
    llvm::errs() << "[" << DEBUG_TYPE << "] " << kVariantCountEnvVar << "='"
                 << env << "' is not a positive number; ignored\n";
    return 1;
  }
  return value;
}

/// Swallows everything written to stdout and stderr while it is alive.
///
/// A trial run drives the whole pipeline, and the pipeline is talkative: debug
/// prints from several passes, MLIR diagnostics from the ones that decline a
/// candidate, and the estimate's own report. Multiplied by the number of
/// candidates that is thousands of lines describing IR that is about to be
/// thrown away. Silencing at the file descriptor rather than by asking each
/// pass to be quiet is the only thing that covers all of them, including
/// prints this pass does not own.
class OutputSilencer {
public:
  OutputSilencer() {
#ifndef _WIN32
    flushAll();
    devNull = ::open("/dev/null", O_WRONLY);
    if (devNull < 0) {
      return; // cannot silence; noisy is better than broken
    }
    savedOut = ::dup(STDOUT_FILENO);
    savedErr = ::dup(STDERR_FILENO);
    ::dup2(devNull, STDOUT_FILENO);
    ::dup2(devNull, STDERR_FILENO);
#endif
  }

  ~OutputSilencer() {
#ifndef _WIN32
    flushAll();
    if (savedOut >= 0) {
      ::dup2(savedOut, STDOUT_FILENO);
      ::close(savedOut);
    }
    if (savedErr >= 0) {
      ::dup2(savedErr, STDERR_FILENO);
      ::close(savedErr);
    }
    if (devNull >= 0) {
      ::close(devNull);
    }
#endif
  }

  OutputSilencer(const OutputSilencer &) = delete;
  OutputSilencer &operator=(const OutputSilencer &) = delete;

private:
  static void flushAll() {
    // Buffered streams must be emptied on both sides of the swap, or their
    // contents surface attributed to the wrong descriptor.
    std::cout.flush();
    std::cerr.flush();
    llvm::outs().flush();
    llvm::errs().flush();
    std::fflush(stdout);
    std::fflush(stderr);
  }

  int devNull = -1;
  int savedOut = -1;
  int savedErr = -1;
};

void restoreModuleFromBackup(ModuleOp moduleOp, ModuleOp moduleBackup) {
  Operation *moduleOperation = moduleOp.getOperation();
  Operation *backupOperation = moduleBackup.getOperation();

  moduleOperation->setLoc(backupOperation->getLoc());
  moduleOperation->setAttrs(backupOperation->getAttrs());
  if (moduleOperation->getPropertiesStorageSize() != 0) {
    moduleOperation->copyProperties(backupOperation->getPropertiesStorage());
  }
  moduleOp.getBodyRegion().takeBody(moduleBackup.getBodyRegion());
}

} // namespace

AddDynamicCVPipelinePass::AddDynamicCVPipelinePass(
    const AddDynamicCVPipelineOptions &options)
    : AddDynamicCVPipelineBase(options) {}

void AddDynamicCVPipelinePass::runOnOperation() {
  auto moduleOp = getOperation();
  OpBuilder builder(moduleOp.getContext());
  compileOn91095Flag = this->compileOn91095;

  LDBG("Enter pass");
  moduleOp->removeAttr(CVPipeline::ERRCODE_ATTR);

  if (!compileOn91095Flag) {
    llvm::errs() << "Add-dynamic-cv-pipeline is only supported on 91095 now.\n";
    return;
  }

  ModuleOp moduleBackup(moduleOp->clone());

  auto buildPipeline = [&](PassManager &pm) {
    pm.addPass(createPreCheckAvailablePass());
    pm.addPass(createStandardizeOpPass());
    pm.addPass(createPlanComputeBlockPass());
    pm.addPass(createComputeBlockOptPass());
    // Unroll the main loop once the compute blocks are planned but before the
    // dataflow is split, so that the inter core transfers, their sync flags and
    // the multi buffers below are planned for each unrolled copy separately.
    // The pass is a no-op unless a factor > 1 was requested.
    if (this->mainLoopUnrollFactor > 1) {
      MainLoopUnrollOptions unrollOptions;
      unrollOptions.unrollFactor = this->mainLoopUnrollFactor;
      pm.addPass(createMainLoopUnrollPass(unrollOptions));
    }

    pm.addPass(createSplitDataflowPass());
    pm.addPass(createAnalyzeDataFlowPass());
    pm.addPass(createAllocMultiCachePass());
    pm.addPass(createAddControlFlowConditionPass());
    pm.addPass(createSeparateMemoryFromComputePass());
    // Must precede createRemoveSsbufAttrPass(): the estimate is driven by the
    // ssbuffer.* attributes that pass strips.
    pm.addPass(createEstimateCVPipelineCostPass());
    pm.addPass(createRemoveSsbufAttrPass());
  };

  // Try several orderings and keep the cheapest, when asked to. The winning
  // seed is compiled once more at the end with the estimate allowed to speak,
  // so the module that survives and the report that describes it are the same
  // compilation rather than two that happen to agree.
  const int variantCount = getVariantCount();
  if (variantCount > 1) {
    int64_t bestSeed = -1;
    int64_t bestCost = 0;
    // The bound the total did not come from, for the winner. Used only to
    // separate candidates whose totals tie.
    int64_t bestSecond = 0;
    int64_t worstCost = 0;
    int64_t usable = 0;
    // How often the second bound was what decided. A search that never used it
    // was ordered by the total alone; one that used it often is running against
    // a wall the total cannot see past, which is worth knowing before trusting
    // the winner.
    int64_t tieBreaks = 0;
    // How many different scores the candidates got. One means every ordering
    // that compiled looked identical to the estimate, which says where to look
    // next: either the generator is producing one shape under many names, or
    // the estimate cannot see what separates them. Without this the two are
    // indistinguishable from outside.
    //
    // Keyed on the pair, not on the total: since the second bound decides
    // ties, two candidates with the same total are no longer the same
    // candidate, and counting totals alone under-reports how much the search
    // can actually tell apart.
    llvm::DenseSet<std::pair<int64_t, int64_t>> distinctCosts;
    // Candidates whose total was indistinguishable from the best so far. The
    // difference between "no ties happened" and "ties happened and none of
    // them was an improvement" is not visible from the winner alone, and the
    // two mean opposite things about whether the second bound is earning its
    // keep.
    int64_t tiedCandidates = 0;
    // Every accepted candidate, so that a winner rejected further down the
    // toolchain -- by the binary compiler running out of Unified Buffer, say,
    // which no pass here can foresee -- leaves the next choices on record
    // instead of sending the search back to the start.
    SmallVector<std::tuple<int64_t, int64_t, int64_t, int64_t>> ranked;
    const std::optional<int64_t> ubSlack = getUBSlackBytes();
    const std::optional<int64_t> stageSlack = getStageSlack();
    // Set by seed 0, the untouched pipeline, which the loop below reaches
    // first. Everything after it is judged against these.
    std::optional<int64_t> baselineUBPeak;
    std::optional<int64_t> baselineDepth;
    // Derived from the baseline when TRITON_ASCEND_CV_UB_SLACK_BYTES says
    // nothing: the room to spend is what the baseline leaves unused.
    int64_t ubAllowance = 0;
    llvm::errs() << "[" << DEBUG_TYPE << "] variant search: trying "
                 << variantCount << " operation orderings\n";

    // Why the rejected candidates were rejected, counted by error code. A
    // search that keeps nothing is otherwise indistinguishable from one that
    // was never asked to do anything.
    SmallVector<std::pair<int, int64_t>> failures;
    auto countFailure = [&](int code) {
      for (auto &entry : failures) {
        if (entry.first == code) {
          ++entry.second;
          return;
        }
      }
      failures.push_back({code, 1});
    };

    for (int64_t seed = 0; seed < variantCount; ++seed) {
      bool accepted = false;
      int64_t cost = 0;
      int64_t second = 0;
      int64_t ubBytes = 0;
      int64_t ubCapacity = 0;
      int64_t depth = 0;
      int errCode = 0;

      {
        // Nothing the trial prints is worth reading: it describes IR that is
        // about to be discarded. The estimate is asked to stay quiet, and the
        // rest of the pipeline is silenced at the descriptor.
        OutputSilencer hush;

        // Every attempt starts from the untouched input: the pipeline rewrites
        // the module in place, so a candidate must not be built on the
        // previous one's output.
        ModuleOp attempt(moduleBackup->clone());
        restoreModuleFromBackup(moduleOp, attempt);
        attempt->destroy();

        moduleOp->setAttr(CVPipeline::kReorderSeed,
                          builder.getI64IntegerAttr(seed));
        moduleOp->setAttr(mlir::triton::kCVPipelineCostQuiet,
                          builder.getUnitAttr());

        PassManager trial(&getContext(), moduleOp.getOperationName());
        buildPipeline(trial);
        const bool ran = !failed(runPipeline(trial, moduleOp)) &&
                         !CVPipeline::hasFallbackAttr(moduleOp);
        auto costAttr = moduleOp->getAttrOfType<IntegerAttr>(
            mlir::triton::kCVPipelineEstimatedCycles);
        if (ran && costAttr) {
          accepted = true;
          cost = costAttr.getInt();
          auto resourceAttr = moduleOp->getAttrOfType<IntegerAttr>(
              mlir::triton::kCVPipelineCostResource);
          auto recurrenceAttr = moduleOp->getAttrOfType<IntegerAttr>(
              mlir::triton::kCVPipelineCostRecurrence);
          // Falling back to the total itself makes every tie compare equal,
          // which is exactly the behaviour before the second bound existed.
          second = resourceAttr && recurrenceAttr
                       ? std::min(resourceAttr.getInt(), recurrenceAttr.getInt())
                       : cost;
          if (auto ubAttr = moduleOp->getAttrOfType<IntegerAttr>(
                  mlir::triton::kCVPipelineCostUBPeak)) {
            ubBytes = ubAttr.getInt();
          }
          // -3: it compiled and scored, but holds more Unified Buffer live than
          // the baseline plus what the caller allowed. Counted as a rejection
          // rather than ranked, so the summary shows how much of the search
          // this threw away -- and if that is most of it, the answer for this
          // kernel is that the partition axis has no room, which is worth
          // seeing rather than inferring from a winner that will not build.
          if (auto capAttr = moduleOp->getAttrOfType<IntegerAttr>(
                  mlir::triton::kCVPipelineCostUBCapacity)) {
            ubCapacity = capAttr.getInt();
          }
          if (auto depthAttr = moduleOp->getAttrOfType<IntegerAttr>(
                  mlir::triton::kCVPipelineCostPipelineDepth)) {
            depth = depthAttr.getInt();
          }
          // -3: it holds more Unified Buffer live than the baseline plus the
          // room the baseline left unused. Both guards below run against the
          // baseline rather than against an absolute, so nothing here is fitted
          // to a kernel; the baseline itself always passes, having set the
          // reference, so the worst a wrong estimate can do is return it.
          if (baselineUBPeak && ubBytes > *baselineUBPeak + ubAllowance) {
            accepted = false;
            errCode = -3;
          }
          // -4: the software pipeline was cut into more stages than the
          // baseline's, which the binary compiler refuses as a prologue that
          // reads a buffer before its first write.
          if (accepted && baselineDepth &&
              depth > *baselineDepth + stageSlack.value_or(0)) {
            accepted = false;
            errCode = -4;
          }
        } else if (!ran) {
          auto codeAttr =
              moduleOp->getAttrOfType<IntegerAttr>(CVPipeline::ERRCODE_ATTR);
          // -1: the pipeline declined without saying why.
          errCode = codeAttr ? static_cast<int>(codeAttr.getInt()) : -1;
        } else {
          // -2: it compiled, but produced no estimate to rank it by.
          errCode = -2;
        }
      }

      if (!accepted) {
        countFailure(errCode);
        continue;
      }

      ++usable;
      if (seed == 0) {
        baselineUBPeak = ubBytes;
        baselineDepth = depth;
        // What the baseline leaves unused, unless the caller named a figure.
        // A baseline that already fills the buffer yields zero and admits only
        // candidates that hold no more than it does -- the right answer, and
        // the one flash attention gives at 128x128.
        ubAllowance = ubSlack ? *ubSlack
                              : std::max<int64_t>(0, ubCapacity - ubBytes);
        llvm::errs() << "[" << DEBUG_TYPE << "]   baseline: " << ubBytes
                     << " bytes of unified buffer live against a capacity of "
                     << ubCapacity << ", pipeline depth " << depth
                     << "; candidates may spend " << ubAllowance
                     << " more byte(s) and " << stageSlack.value_or(0)
                     << " more stage(s)\n";
      }
      distinctCosts.insert({cost, second});
      ranked.push_back({cost, second, ubBytes, seed});
      if (worstCost < cost) {
        worstCost = cost;
      }
      // Order by the total, and by the other bound only when the totals cannot
      // be told apart. The two bounds are not interchangeable -- the total is
      // still what the model claims the kernel costs -- so this never lets a
      // candidate with a worse total win.
      bool improves = bestSeed < 0;
      bool byTieBreak = false;
      if (!improves) {
        if (totalsAreTied(cost, bestCost)) {
          ++tiedCandidates;
          improves = second < bestSecond;
          byTieBreak = improves;
        } else {
          improves = cost < bestCost;
        }
      }
      if (improves) {
        bestSeed = seed;
        bestCost = cost;
        bestSecond = second;
        if (byTieBreak) {
          ++tieBreaks;
        }
        llvm::errs() << "[" << DEBUG_TYPE << "]   seed " << seed << ": " << cost
                     << " cycles";
        if (byTieBreak) {
          llvm::errs() << " (tied on the total; won on the other bound at "
                       << second << ")";
        }
        llvm::errs() << " (best so far)\n";
      }
    }

    llvm::errs() << "[" << DEBUG_TYPE << "] variant search: " << usable << " of "
                 << variantCount << " ordering(s) compiled";
    if (bestSeed >= 0) {
      llvm::errs() << ", keeping seed " << bestSeed << " at " << bestCost
                   << " cycles\n";
      llvm::errs() << "[" << DEBUG_TYPE << "]   " << distinctCosts.size()
                   << " distinct estimate(s), worst " << worstCost
                   << " cycles; seed 0 is the untouched pipeline\n";
      if (tiedCandidates > 0) {
        llvm::errs() << "[" << DEBUG_TYPE << "]   " << tiedCandidates
                     << " candidate(s) tied with the best on the total, "
                     << tieBreaks << " of which won on the second bound";
        if (tieBreaks == 0) {
          llvm::errs() << " -- so the ties were real and none of them had more"
                          " room than the incumbent, which is what to expect"
                          " when the best candidate already has the shortest"
                          " dependency chain";
        }
        llvm::errs() << "\n";
      }
      // Ordered the same way the search ordered them, so the runner-up is a
      // seed and not a re-run. Worth having because a winner can still be
      // refused by the binary compiler for something no pass here models --
      // Unified Buffer capacity being the case that has actually bitten.
      // One seed per distinct score. Many seeds reach the same pair -- 118 of
      // 200 tied on one run -- and listing five of those is five names for one
      // candidate, which is useless precisely when the list is needed: the
      // winner having been refused downstream, what is wanted is a *different*
      // candidate to try, not another spelling of the same one.
      llvm::sort(ranked);
      llvm::errs() << "[" << DEBUG_TYPE
                   << "]   best distinct score(s), in case the winner is"
                      " refused downstream:";
      int64_t shown = 0;
      for (size_t i = 0; i < ranked.size() && shown < 5; ++i) {
        if (i > 0 && std::get<0>(ranked[i]) == std::get<0>(ranked[i - 1]) &&
            std::get<1>(ranked[i]) == std::get<1>(ranked[i - 1])) {
          continue;
        }
        llvm::errs() << " " << std::get<3>(ranked[i]) << "("
                     << std::get<0>(ranked[i]) << "/" << std::get<1>(ranked[i])
                     << "/" << std::get<2>(ranked[i]) << "B)";
        ++shown;
      }
      llvm::errs() << "  [seed(total/second bound/unified buffer)]\n";
    } else {
      llvm::errs() << ", none usable; compiling without a variant\n";
    }
    for (const auto &entry : failures) {
      llvm::errs() << "[" << DEBUG_TYPE << "]   rejected " << entry.second
                   << " with ";
      if (entry.first == -1) {
        llvm::errs() << "a pipeline failure and no error code\n";
      } else if (entry.first == -2) {
        llvm::errs() << "no estimate produced\n";
      } else if (entry.first == -3) {
        llvm::errs() << "more than the baseline's "
                     << baselineUBPeak.value_or(0) << " bytes of unified"
                        " buffer plus its "
                     << ubAllowance << "-byte allowance\n";
      } else if (entry.first == -4) {
        llvm::errs() << "a deeper software pipeline than the baseline's "
                     << baselineDepth.value_or(0) << " stage(s)\n";
      } else {
        llvm::errs() << "error code " << entry.first << "\n";
      }
    }
    if (bestSeed < 0) {
      llvm::errs() << "[" << DEBUG_TYPE
                   << "]   reproduce one of them with"
                      " TRITON_ASCEND_REORDER_SEED=0 and no"
                      " TRITON_ASCEND_CV_VARIANTS to see the failure itself\n";
    }

    // Reset to the input and compile the winner for real, with the estimate
    // allowed to report. A search that found nothing leaves the seed unset,
    // which is the ordinary path.
    ModuleOp fresh(moduleBackup->clone());
    restoreModuleFromBackup(moduleOp, fresh);
    fresh->destroy();
    if (bestSeed >= 0) {
      moduleOp->setAttr(CVPipeline::kReorderSeed,
                        builder.getI64IntegerAttr(bestSeed));
    }
  }

  PassManager pm(&getContext(), moduleOp.getOperationName());
  buildPipeline(pm);

  if (failed(runPipeline(pm, moduleOp)) ||
      CVPipeline::hasFallbackAttr(moduleOp)) {
    auto errCodeAttr =
        moduleOp->getAttrOfType<IntegerAttr>(CVPipeline::ERRCODE_ATTR);
    if (!errCodeAttr) {
      moduleOp->emitWarning() << "[" << DEBUG_TYPE << "] "
                              << "Unexpected pass failure (no fallback attr "
                                 "set); fallback to compilation without "
                                 "dynamic CV pipeline.";
    } else {
      moduleOp->emitWarning() << "[" << DEBUG_TYPE << "] "
                              << "Pass failed, "
                              << "fallback to compilation without "
                                 "dynamic CV pipeline.";
    }

    int errCode = errCodeAttr ? static_cast<int>(errCodeAttr.getInt())
                              : CVPipeline::ERRCODE_FAILED;
    std::cout << "[VDV DEBUG] DynamicCVPipeline failed errCode=" << errCode << std::endl;
    restoreModuleFromBackup(moduleOp, moduleBackup);
    moduleBackup->destroy();
    moduleOp->setAttr(CVPipeline::ERRCODE_ATTR,
                      builder.getI32IntegerAttr(errCode));
    return;
  }

  moduleBackup->destroy();
  LDBG("Process successfully");
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createAddDynamicCVPipelinePass(
    const AddDynamicCVPipelineOptions &options) {
  return std::make_unique<AddDynamicCVPipelinePass>(options);
}
