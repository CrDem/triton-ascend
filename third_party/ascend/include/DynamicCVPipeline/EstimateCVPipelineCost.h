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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ESTIMATE_CV_PIPELINE_COST_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ESTIMATE_CV_PIPELINE_COST_H

#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace triton {

/// Estimated cost of the CV-pipelined module, in hardware cycles (i64).
/// Consumers rank alternative CV-pipeline IR variants by this number: lower is
/// better. Absent when the costmodel is not built into this binary, or when the
/// estimate could not be produced.
inline constexpr llvm::StringLiteral kCVPipelineEstimatedCycles =
    "ascend.cv_pipeline_estimated_cycles";

/// Name of the hardware profile the estimate was produced against (StringAttr).
/// Estimates are only comparable across modules sharing the same profile.
inline constexpr llvm::StringLiteral kCVPipelineCostHardware =
    "ascend.cv_pipeline_cost_hardware";

/// Number of operations whose cost could not be determined (i64), e.g. dynamic
/// shapes or loops with a non-constant trip count. A high count relative to the
/// module size means the estimate should not be trusted.
inline constexpr llvm::StringLiteral kCVPipelineCostUnknownOps =
    "ascend.cv_pipeline_cost_unknown_ops";

/// Estimates the execution cost of the dynamic CV pipeline IR and records it as
/// module attributes. Purely advisory: it never mutates compute IR and never
/// fails the pipeline, so a costmodel problem can not break compilation.
///
/// Must run before RemoveSsbufAttrPass, which strips the ssbuffer.* attributes
/// (notably ssbuffer.core_type) this pass reads.
class EstimateCVPipelineCostPass
    : public PassWrapper<EstimateCVPipelineCostPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EstimateCVPipelineCostPass);

  EstimateCVPipelineCostPass() = default;
  explicit EstimateCVPipelineCostPass(std::string hardwareConfigPath)
      : hardwareConfigPath(std::move(hardwareConfigPath)) {}

  void runOnOperation() override;

  llvm::StringRef getArgument() const final {
    return "estimate-cv-pipeline-cost";
  }

private:
  /// Path to a costmodel hardware JSON. Empty selects the built-in default.
  std::string hardwareConfigPath;
};

std::unique_ptr<OperationPass<ModuleOp>>
createEstimateCVPipelineCostPass(std::string hardwareConfigPath = "");
void registerEstimateCVPipelineCostPasses();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ESTIMATE_CV_PIPELINE_COST_H
