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
    // How many different numbers the candidates scored. One means every
    // ordering that compiled looked identical to the estimate, which says
    // where to look next: either the generator is producing one shape under
    // many names, or the estimate cannot see what separates them. Without this
    // the two are indistinguishable from outside.
    llvm::DenseSet<int64_t> distinctCosts;
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
      distinctCosts.insert(cost);
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
      if (tieBreaks > 0) {
        llvm::errs() << "[" << DEBUG_TYPE << "]   " << tieBreaks
                     << " improvement(s) came from the second bound after the"
                        " totals tied -- the winner is held back by the same"
                        " constraint as the rest, and was chosen for having"
                        " more room before the other one bites\n";
      }
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
