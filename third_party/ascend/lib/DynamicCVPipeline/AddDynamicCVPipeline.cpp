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

#include <cstdlib>
#include <iostream>

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
    int64_t usable = 0;
    llvm::errs() << "[" << DEBUG_TYPE << "] variant search: trying "
                 << variantCount << " operation orderings\n";

    for (int64_t seed = 0; seed < variantCount; ++seed) {
      // Every attempt starts from the untouched input: the pipeline rewrites
      // the module in place, so a candidate must not be built on the previous
      // one's output.
      ModuleOp attempt(moduleBackup->clone());
      restoreModuleFromBackup(moduleOp, attempt);
      attempt->destroy();

      moduleOp->setAttr(CVPipeline::kReorderSeed,
                        builder.getI64IntegerAttr(seed));
      moduleOp->setAttr(mlir::triton::kCVPipelineCostQuiet,
                        builder.getUnitAttr());

      PassManager trial(&getContext(), moduleOp.getOperationName());
      buildPipeline(trial);
      if (failed(runPipeline(trial, moduleOp)) ||
          CVPipeline::hasFallbackAttr(moduleOp)) {
        continue; // this ordering broke something downstream; not a candidate
      }
      auto costAttr =
          moduleOp->getAttrOfType<IntegerAttr>(
              mlir::triton::kCVPipelineEstimatedCycles);
      if (!costAttr) {
        continue; // no estimate, so nothing to rank it by
      }

      ++usable;
      const int64_t cost = costAttr.getInt();
      if (bestSeed < 0 || cost < bestCost) {
        bestSeed = seed;
        bestCost = cost;
        llvm::errs() << "[" << DEBUG_TYPE << "]   seed " << seed << ": " << cost
                     << " cycles (best so far)\n";
      }
    }

    llvm::errs() << "[" << DEBUG_TYPE << "] variant search: " << usable << " of "
                 << variantCount << " ordering(s) compiled";
    if (bestSeed >= 0) {
      llvm::errs() << ", keeping seed " << bestSeed << " at " << bestCost
                   << " cycles\n";
    } else {
      llvm::errs() << ", none usable; compiling without a variant\n";
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
