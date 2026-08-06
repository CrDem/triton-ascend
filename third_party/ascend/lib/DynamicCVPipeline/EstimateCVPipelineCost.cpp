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

//===----------------------------------------------------------------------===//
//
// Estimates the execution cost of the IR produced by the dynamic CV pipeline.
//
// Unlike the standalone costmodel pipeline (-ascend-perf-model), which runs on
// Triton IR and has to *guess* what the compiler will do, this pass runs at the
// end of the CV pipeline and reads the decisions the compiler actually made:
// the Cube/Vector split is taken from the ssbuffer.core_type attributes stamped
// by OpClassifierPass rather than re-derived. Only the hardware cost formulas
// (HardwareConfig) and the roofline combination (PipelineScheduler) are reused
// from the costmodel.
//
// Scope of this version: per-operation costs are accumulated per hardware unit
// and combined with the roofline model. Data dependencies between operations
// are NOT modelled yet, so the result is a lower bound on the critical path
// rather than a schedule. Multi-buffering / cross-core sync are likewise not
// modelled. The number is intended for *ranking* CV pipeline variants against
// each other, not for absolute latency prediction.
//
// Because a single number hides how much of it was actually modelled, every
// operation is classified (see CostConfidence) and a per-operation-kind
// breakdown can be printed. Operations charged generically are the ones worth
// teaching the model about next; operations of unknown size contribute nothing
// at all and would otherwise be invisible.
//
//   TRITON_ASCEND_CV_COST_VERBOSE=1   one-line summary
//   TRITON_ASCEND_CV_COST_VERBOSE=2   summary + per-operation breakdown
//
//===----------------------------------------------------------------------===//

#include "ascend/include/DynamicCVPipeline/EstimateCVPipelineCost.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <utility>

#if TRITON_ASCEND_HAS_INPROC_COSTMODEL
#include "AscendModel/Analysis/HardwareConfig.h"
#include "AscendModel/Analysis/PipelineAnalysis.h"
#include "AscendModel/Analysis/Utils.h"
#endif

using namespace mlir;
using namespace mlir::triton;

static constexpr const char *DEBUG_TYPE = "estimate-cv-pipeline-cost";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LOG_DEBUG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

#if TRITON_ASCEND_HAS_INPROC_COSTMODEL

namespace {

/// Opt-in stderr summary. LLVM_DEBUG requires a debug build plus -debug-only,
/// which is impractical when driving compilation from Python; this env var
/// gives the same one-line summary from a release build.
constexpr const char *kVerboseEnvVar = "TRITON_ASCEND_CV_COST_VERBOSE";

using mlir::ascend::HardwareConfig;
using mlir::ascend::HWUnit;

//===----------------------------------------------------------------------===//
// Shape helpers
//===----------------------------------------------------------------------===//
// The CV pipeline operates on partially bufferized IR: the same logical buffer
// appears as a tensor before bufferization and as a memref after. Both are
// ShapedType, so cost extraction is written against that rather than against
// RankedTensorType (which is all the costmodel's own helpers handle).

/// Number of elements, or nullopt when the shape is not statically known.
std::optional<int64_t> getShapedElementCount(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape()) {
    return std::nullopt;
  }
  int64_t count = 1;
  for (int64_t dim : shaped.getShape()) {
    count *= dim;
  }
  return count;
}

int getShapedElementBits(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  Type elemType = shaped ? shaped.getElementType() : type;
  if (elemType.isIntOrFloat()) {
    return elemType.getIntOrFloatBitWidth();
  }
  return 32; // conservative default for opaque element types
}

/// Byte size, or nullopt when the shape is not statically known.
std::optional<int64_t> getShapedByteSize(Type type) {
  auto count = getShapedElementCount(type);
  if (!count) {
    return std::nullopt;
  }
  return *count * ((getShapedElementBits(type) + 7) / 8);
}

/// First operand or result carrying a shaped type, used to size elementwise
/// ops whose cost is driven by their working-set size.
Type getRepresentativeShapedType(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (isa<ShapedType>(operand.getType())) {
      return operand.getType();
    }
  }
  for (Value result : op->getResults()) {
    if (isa<ShapedType>(result.getType())) {
      return result.getType();
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Core-type resolution
//===----------------------------------------------------------------------===//

/// Resolve whether an op runs on the Cube core. Primary source is the
/// ssbuffer.core_type attribute stamped by OpClassifierPass; ops that never got
/// stamped (or carry a multi-valued type) fall back to the tcore_type of the
/// enclosing scope::ScopeOp created by SeparateCVScopePass.
bool runsOnCubeCore(Operation *op) {
  switch (CVPipeline::getOpCoreType(op)) {
  case CVPipeline::CoreType::CUBE_ONLY:
    return true;
  case CVPipeline::CoreType::VECTOR_ONLY:
    return false;
  default:
    break;
  }

  if (auto scopeOp = op->getParentOfType<scope::ScopeOp>()) {
    bool isCube = false;
    bool isVector = false;
    if (succeeded(triton::getScopeType(scopeOp, isCube, isVector)) && isCube &&
        !isVector) {
      return true;
    }
  }
  return false; // Vector is the default core, matching OpClassifierPass.
}

//===----------------------------------------------------------------------===//
// Operation classification
//===----------------------------------------------------------------------===//

/// Ops that carry no hardware cost: structure, metadata, addressing and scalar
/// bookkeeping. Address arithmetic is absorbed into DMA descriptors on Ascend,
/// and allocations are compile-time placement rather than runtime work.
bool isZeroCostOp(Operation *op) {
  if (isa<ModuleOp, func::FuncOp, func::ReturnOp, func::CallOp>(op)) {
    return true;
  }
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return true; // yields/returns transfer values, they do not compute
  }
  // A linalg body describes the elementwise kernel of its parent op; the parent
  // is costed as a whole, so charging its body again would double-count.
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<linalg::LinalgDialect>(parent->getDialect())) {
      return true;
    }
  }
  if (CVPipeline::isScfOp(op)) {
    return true; // loop/branch structure; the body is costed on its own
  }
  if (isa<scope::ScopeOp>(op)) {
    return true; // container introduced by SeparateCVScopePass
  }
  if (isa<annotation::MarkOp>(op)) {
    return true; // annotation only
  }
  if (isa<memref::AllocOp, memref::AllocaOp, memref::DeallocOp,
          tensor::EmptyOp, bufferization::AllocTensorOp,
          bufferization::ToTensorOp>(op)) {
    return true; // buffer placement / tensor-memref bridging
  }
  if (CVPipeline::isViewLike(op)) {
    return true; // subview/reshape: descriptor math, no data movement
  }
  if (isa<arith::ConstantOp>(op)) {
    return true;
  }
  // Scalar arith/math (loop bounds, offsets, predicates) is bookkeeping; only
  // shaped operations occupy a compute pipe.
  if (isa<arith::ArithDialect, math::MathDialect>(op->getDialect())) {
    return !getRepresentativeShapedType(op);
  }
  return false;
}

/// Whether a value's backing buffer is a local (on-chip) allocation, as opposed
/// to a kernel argument living in global memory. Mirrors the dest-tracing rule
/// MarkGMLoadPass uses to recognise GM loads.
bool isBackedByLocalAlloc(Value value) {
  Value current = value;
  while (Operation *def = current.getDefiningOp()) {
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(def)) {
      current = viewLike.getViewSource();
      continue;
    }
    if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(def)) {
      current = extractSlice.getSource();
      continue;
    }
    break;
  }
  Operation *def = current.getDefiningOp();
  return def && isa<memref::AllocOp, memref::AllocaOp>(def);
}

/// Hardware unit a data transfer occupies, given its direction and core.
/// Ascend keeps separate movers per core: the Cube path loads through MTE2 into
/// L1 and writes back through FixPipe, the Vector path loads through its own
/// MTE2 into UB and writes back through MTE3.
HWUnit getTransferUnit(bool isCube, bool isLoad) {
  if (isCube) {
    return isLoad ? HWUnit::CubeMTE2 : HWUnit::FixPipe;
  }
  return isLoad ? HWUnit::VecMTE2 : HWUnit::MTE3;
}

/// Memory spaces a transfer moves between, used to select the measured
/// per-(src,dst) bandwidth table entry.
std::pair<llvm::StringRef, llvm::StringRef> getTransferSpaces(HWUnit unit) {
  switch (unit) {
  case HWUnit::CubeMTE2:
    return {"hbm", "l1"};
  case HWUnit::FixPipe:
    return {"l0c", "hbm"};
  case HWUnit::VecMTE2:
    return {"hbm", "ub"};
  case HWUnit::MTE3:
    return {"ub", "hbm"};
  default:
    return {"hbm", "ub"};
  }
}

int64_t getTransferStartupLatency(HWUnit unit, const HardwareConfig &config) {
  switch (unit) {
  case HWUnit::CubeMTE2:
  case HWUnit::VecMTE2:
    return config.getMTE2StartupLatency();
  case HWUnit::FixPipe:
    return config.getFixPipeStartupLatency();
  case HWUnit::MTE3:
    return config.getMTE3StartupLatency();
  default:
    return 0;
  }
}

//===----------------------------------------------------------------------===//
// Cost estimation
//===----------------------------------------------------------------------===//

/// How much the model actually knows about an operation, as opposed to how
/// confident the resulting number looks. Reported per operation kind so that
/// the list of things worth teaching the model is visible rather than guessed.
enum class CostConfidence {
  /// The operation kind has a dedicated cost model and every input that model
  /// needs (shape, dtype, sizes) is statically known.
  Modelled,
  /// No dedicated model for this operation kind: it was charged generically
  /// from its element count. The number is an order of magnitude, not a model.
  Generic,
  /// A required input is not statically known -- a dynamic shape -- so no cost
  /// could be computed at all and the operation contributes nothing.
  UnknownSize,
};

llvm::StringRef stringifyConfidence(CostConfidence confidence) {
  switch (confidence) {
  case CostConfidence::Modelled:
    return "modelled";
  case CostConfidence::Generic:
    return "generic";
  case CostConfidence::UnknownSize:
    return "unknown-size";
  }
  return "?";
}

struct OpCost {
  HWUnit unit = HWUnit::Scalar;
  int64_t cycles = 0;
  int64_t bytes = 0;
  int64_t flops = 0;
  CostConfidence confidence = CostConfidence::Modelled;
};

/// Cost of a linalg.matmul on the Cube core.
std::optional<OpCost> estimateMatmul(linalg::MatmulOp matmulOp,
                                     const HardwareConfig &config) {
  OpCost cost;
  cost.unit = HWUnit::Cube;

  auto inputs = matmulOp.getInputs();
  if (inputs.size() < 2) {
    cost.confidence = CostConfidence::UnknownSize;
    return cost;
  }
  auto lhsType = dyn_cast<ShapedType>(inputs[0].getType());
  auto rhsType = dyn_cast<ShapedType>(inputs[1].getType());
  if (!lhsType || !rhsType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape()) {
    cost.confidence = CostConfidence::UnknownSize;
    return cost;
  }

  int64_t m = lhsType.getShape()[0];
  int64_t k = lhsType.getShape()[1];
  int64_t n = rhsType.getShape()[1];

  // TODO: switch to the tilesim-migrated L1-pipe model used by
  // ascend::MatmulOp::estimateCycles; that formula is currently tied to the
  // costmodel's own dialect op and needs extracting before it can be reused.
  cost.cycles =
      config.estimateCubeCycles(m, n, k) + config.getCubeStartupLatency();
  cost.flops = 2 * m * n * k;
  return cost;
}

/// Cost of a bulk data transfer (memref.copy / hivm store).
OpCost estimateTransfer(Type shapedType, bool isCube, bool isLoad,
                        const HardwareConfig &config) {
  OpCost cost;
  cost.unit = getTransferUnit(isCube, isLoad);

  auto bytes = getShapedByteSize(shapedType);
  if (!bytes) {
    cost.confidence = CostConfidence::UnknownSize;
    return cost;
  }
  cost.bytes = *bytes;

  auto [srcSpace, dstSpace] = getTransferSpaces(cost.unit);
  int64_t transferCycles = config.estimateTransferCycles(
      srcSpace, dstSpace, *bytes, config.getActiveBandwidthCores());
  cost.cycles = transferCycles + getTransferStartupLatency(cost.unit, config);
  return cost;
}

/// Cost of an elementwise / reduction style compute op on the Vector core.
OpCost estimateVectorCompute(Operation *op, const HardwareConfig &config) {
  OpCost cost;
  cost.unit = HWUnit::Vector;

  Type shapedType = getRepresentativeShapedType(op);
  if (!shapedType) {
    cost.confidence = CostConfidence::UnknownSize;
    return cost;
  }
  auto elements = getShapedElementCount(shapedType);
  if (!elements) {
    cost.confidence = CostConfidence::UnknownSize;
    return cost;
  }

  // Charged purely by element count: this path has no knowledge of which
  // vector instruction the operation lowers to, so every op of a given size
  // costs the same. Flagged as Generic precisely so the operations that end up
  // here show up in the breakdown as candidates for a dedicated model.
  cost.cycles =
      config.estimateVectorCycles(*elements) + config.getVectorStartupLatency();
  cost.flops = *elements;
  cost.confidence = CostConfidence::Generic;
  return cost;
}

/// Classify an operation and estimate what it costs. Returns nullopt for ops
/// that do not occupy any hardware pipe.
std::optional<OpCost> estimateOpCost(Operation *op,
                                     const HardwareConfig &config) {
  if (isZeroCostOp(op)) {
    return std::nullopt;
  }

  const bool isCube = runsOnCubeCore(op);

  if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
    return estimateMatmul(matmulOp, config);
  }

  if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
    // A copy into a local allocation is a load; a copy out of one is a
    // write-back. Local-to-local copies are treated as loads.
    const bool targetIsLocal = isBackedByLocalAlloc(copyOp.getTarget());
    const bool sourceIsLocal = isBackedByLocalAlloc(copyOp.getSource());
    const bool isLoad = targetIsLocal || !sourceIsLocal;
    return estimateTransfer(copyOp.getTarget().getType(), isCube, isLoad,
                            config);
  }

  if (CVPipeline::isStoreLike(op)) {
    Type shapedType = getRepresentativeShapedType(op);
    if (!shapedType) {
      OpCost cost;
      cost.unit = getTransferUnit(isCube, /*isLoad=*/false);
      cost.confidence = CostConfidence::UnknownSize;
      return cost;
    }
    return estimateTransfer(shapedType, isCube, /*isLoad=*/false, config);
  }

  // Remaining shaped compute. Cube-side non-matmul work (e.g. a fill or
  // transpose staged for the Cube pipe) is charged to the Cube unit; everything
  // else is Vector work.
  OpCost cost = estimateVectorCompute(op, config);
  if (isCube) {
    cost.unit = HWUnit::Cube;
  }
  return cost;
}

//===----------------------------------------------------------------------===//
// Loop weighting
//===----------------------------------------------------------------------===//

struct LoopWeight {
  int64_t multiplier = 1;
  bool isKnown = true;
};

/// How many times an operation executes, given the enclosing loop nest.
LoopWeight getLoopWeight(Operation *op) {
  LoopWeight weight;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto forOp = dyn_cast<scf::ForOp>(parent);
    if (!forOp) {
      continue;
    }
    auto tripCount = mlir::ascend::utils::analyzeScfForTripCount(forOp);
    if (tripCount.isStatic) {
      weight.multiplier *= tripCount.staticTripCount;
    } else {
      // Dynamic bound: counting one iteration keeps the estimate finite, but
      // the caller must know the number understates the real cost.
      weight.isKnown = false;
    }
  }
  return weight;
}

//===----------------------------------------------------------------------===//
// Roofline combination
//===----------------------------------------------------------------------===//

/// Combine per-unit busy cycles into a module-level estimate.
///
/// Ascend overlaps independent pipes, so a path costs as much as its busiest
/// unit rather than the sum of its units. The exception is the AIV MTE2/MTE3
/// pair, which shares one physical pipeline and therefore serialises; the
/// hardware profile declares that via its mutex groups. Cube and Vector cores
/// run concurrently, so the module costs as much as the slower of the two.
int64_t combineRoofline(const llvm::DenseMap<HWUnit, int64_t> &unitCycles,
                        const HardwareConfig &config) {
  auto cyclesOf = [&](HWUnit unit) -> int64_t {
    auto it = unitCycles.find(unit);
    return it == unitCycles.end() ? 0 : it->second;
  };

  int64_t cubePathCycles = std::max({cyclesOf(HWUnit::Cube),
                                     cyclesOf(HWUnit::CubeMTE2),
                                     cyclesOf(HWUnit::FixPipe)});

  int64_t vectorTransferCycles =
      config.areMutexUnits("vec_mte2", "mte3")
          ? cyclesOf(HWUnit::VecMTE2) + cyclesOf(HWUnit::MTE3)
          : std::max(cyclesOf(HWUnit::VecMTE2), cyclesOf(HWUnit::MTE3));
  int64_t vectorPathCycles =
      std::max(cyclesOf(HWUnit::Vector), vectorTransferCycles);

  return std::max(cubePathCycles, vectorPathCycles);
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//
// The breakdown exists to answer one question: which operations does the model
// actually understand? Everything charged generically is a candidate for a
// dedicated cost model, and everything with an unknown size silently
// contributes nothing. Both are invisible in the single summary number.

struct OpKindStats {
  int64_t count = 0;
  int64_t weightedCycles = 0;
  HWUnit unit = HWUnit::Scalar;
  CostConfidence confidence = CostConfidence::Modelled;
};

/// Ordering used when several operations of one kind disagree: report the
/// least-known outcome, since that is the one worth acting on.
int confidenceRank(CostConfidence confidence) {
  switch (confidence) {
  case CostConfidence::Modelled:
    return 0;
  case CostConfidence::Generic:
    return 1;
  case CostConfidence::UnknownSize:
    return 2;
  }
  return 0;
}

struct CostBreakdown {
  llvm::MapVector<llvm::StringRef, OpKindStats> byOpKind;
  llvm::SmallVector<Operation *> unknownSizeOps;
  llvm::SmallVector<Operation *> dynamicTripCountOps;
  int64_t genericOps = 0;
  int64_t totalWeightedCycles = 0;

  void record(Operation *op, const OpCost &cost, const LoopWeight &weight) {
    OpKindStats &stats = byOpKind[op->getName().getStringRef()];
    stats.count += 1;
    stats.weightedCycles += cost.cycles * weight.multiplier;
    stats.unit = cost.unit;
    if (confidenceRank(cost.confidence) > confidenceRank(stats.confidence)) {
      stats.confidence = cost.confidence;
    }

    totalWeightedCycles += cost.cycles * weight.multiplier;
    if (cost.confidence == CostConfidence::Generic) {
      ++genericOps;
    }
    if (cost.confidence == CostConfidence::UnknownSize) {
      unknownSizeOps.push_back(op);
    }
    if (!weight.isKnown) {
      dynamicTripCountOps.push_back(op);
    }
  }
};

/// Cap on how many individual operations are listed per category, so a large
/// kernel produces a readable report rather than a wall of text.
constexpr size_t kMaxListedOps = 20;

void printOpList(llvm::raw_ostream &os, llvm::StringRef title,
                 llvm::ArrayRef<Operation *> ops) {
  if (ops.empty()) {
    return;
  }
  os << "[" << DEBUG_TYPE << "] " << ops.size() << " " << title << ":\n";
  for (size_t i = 0; i < ops.size() && i < kMaxListedOps; ++i) {
    os << "[" << DEBUG_TYPE << "]     "
       << ops[i]->getName().getStringRef() << "  at " << ops[i]->getLoc()
       << "\n";
  }
  if (ops.size() > kMaxListedOps) {
    os << "[" << DEBUG_TYPE << "]     ... and " << (ops.size() - kMaxListedOps)
       << " more\n";
  }
}

void printBreakdown(llvm::raw_ostream &os, const CostBreakdown &breakdown) {
  // Heaviest first: that is the order in which teaching the model new
  // operations pays off.
  llvm::SmallVector<std::pair<llvm::StringRef, OpKindStats>> rows;
  for (const auto &entry : breakdown.byOpKind) {
    rows.push_back({entry.first, entry.second});
  }
  llvm::sort(rows, [](const auto &lhs, const auto &rhs) {
    return lhs.second.weightedCycles > rhs.second.weightedCycles;
  });

  const double total =
      breakdown.totalWeightedCycles > 0
          ? static_cast<double>(breakdown.totalWeightedCycles)
          : 1.0;

  os << "[" << DEBUG_TYPE << "] cost breakdown by operation kind"
     << " (share is of total work, not of the estimate):\n";
  os << "[" << DEBUG_TYPE << "]     count       cycles   share  confidence  "
     << "  unit        operation\n";
  for (const auto &[name, stats] : rows) {
    os << "[" << DEBUG_TYPE << "] "
       << llvm::format("%9lld", static_cast<long long>(stats.count))
       << llvm::format("%13lld", static_cast<long long>(stats.weightedCycles))
       << llvm::format("%7.1f%%", 100.0 * stats.weightedCycles / total) << "  "
       << llvm::left_justify(stringifyConfidence(stats.confidence), 12)
       << llvm::left_justify(mlir::ascend::stringifyHWUnit(stats.unit), 12)
       << name << "\n";
  }

  // The actionable lists.
  llvm::SmallVector<llvm::StringRef> genericKinds;
  for (const auto &[name, stats] : rows) {
    if (stats.confidence == CostConfidence::Generic) {
      genericKinds.push_back(name);
    }
  }
  if (!genericKinds.empty()) {
    os << "[" << DEBUG_TYPE << "] " << genericKinds.size()
       << " operation kind(s) charged by element count only, i.e. with no"
          " dedicated cost model:\n";
    for (llvm::StringRef name : genericKinds) {
      os << "[" << DEBUG_TYPE << "]     " << name << "\n";
    }
  }

  printOpList(os, "operation(s) with a non-static shape, contributing 0 cycles",
              breakdown.unknownSizeOps);
  printOpList(os,
              "operation(s) in a loop with a dynamic trip count, counted once",
              breakdown.dynamicTripCountOps);
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Verbosity requested through the environment: 0 = silent, 1 = one-line
/// summary, >=2 = summary plus the per-operation breakdown.
int getVerbosity() {
  const char *raw = std::getenv(kVerboseEnvVar);
  if (!raw) {
    return 0;
  }
  int level = std::atoi(raw);
  // Any non-numeric but present value means "on"; only an explicit 0 is off.
  if (level == 0 && raw[0] != '0') {
    return 1;
  }
  return level;
}

void estimateModuleCost(ModuleOp module, llvm::StringRef hardwareConfigPath) {
  std::string configError;
  auto config = mlir::ascend::loadHardwareConfigForAnalysis(hardwareConfigPath,
                                                            configError);
  if (!config) {
    LOG_DEBUG("failed to load hardware config: " << configError);
    return;
  }

  mlir::ascend::PipelineScheduler scheduler(config.get());
  llvm::DenseMap<HWUnit, int64_t> unitCycles;
  CostBreakdown breakdown;
  int64_t nextOpId = 0;
  int64_t unknownOps = 0;

  module.walk([&](Operation *op) {
    auto cost = estimateOpCost(op, *config);
    if (!cost) {
      return;
    }

    LoopWeight weight = getLoopWeight(op);
    if (cost->confidence == CostConfidence::UnknownSize || !weight.isKnown) {
      ++unknownOps;
    }
    breakdown.record(op, *cost, weight);

    // NOTE: dependencies are intentionally not registered yet, so the
    // scheduler's own critical path is not meaningful here; the reported
    // number comes from the per-unit roofline below. Wiring memory-level
    // dependencies in is what turns this into a real schedule.
    mlir::ascend::PipelineOp pipelineOp;
    pipelineOp.opId = nextOpId++;
    pipelineOp.hwUnit = cost->unit;
    pipelineOp.duration = cost->cycles;
    pipelineOp.bytes = cost->bytes;
    pipelineOp.flops = cost->flops;
    pipelineOp.loopMultiplier = weight.multiplier;
    pipelineOp.mlirOp = op;
    pipelineOp.opName = op->getName().getStringRef().str();
    scheduler.addOperation(pipelineOp);

    unitCycles[cost->unit] += cost->cycles * weight.multiplier;
  });

  if (nextOpId == 0) {
    LOG_DEBUG("no costed operations found; skipping estimate");
    return;
  }

  // Critical path over one iteration. With no dependencies registered this
  // currently degenerates to the busiest unit's serial time; it is reported
  // alongside the roofline so the two can be compared once dependency wiring
  // makes it a real schedule.
  scheduler.schedule();
  int64_t scheduledCycles = scheduler.getTotalCycles();

  int64_t totalCycles = combineRoofline(unitCycles, *config);

  Builder builder(module.getContext());
  module->setAttr(kCVPipelineEstimatedCycles,
                  builder.getI64IntegerAttr(totalCycles));
  module->setAttr(kCVPipelineCostHardware,
                  builder.getStringAttr(config->getName()));
  module->setAttr(kCVPipelineCostUnknownOps,
                  builder.getI64IntegerAttr(unknownOps));
  module->setAttr(kCVPipelineCostGenericOps,
                  builder.getI64IntegerAttr(breakdown.genericOps));

  auto reportTo = [&](llvm::raw_ostream &os) {
    os << "[" << DEBUG_TYPE << "] " << config->getName() << ": " << totalCycles
       << " cycles roofline ("
       << llvm::format("%.3f", config->cyclesToMicroseconds(totalCycles))
       << " us), " << scheduledCycles << " cycles critical path, " << nextOpId
       << " ops costed, " << unknownOps << " of unknown cost, "
       << breakdown.genericOps << " without a dedicated model\n";
  };

  const int verbosity = getVerbosity();
  LLVM_DEBUG(reportTo(llvm::dbgs()); printBreakdown(llvm::dbgs(), breakdown));
  if (verbosity >= 1) {
    reportTo(llvm::errs());
  }
  if (verbosity >= 2) {
    printBreakdown(llvm::errs(), breakdown);
  }
}

} // namespace

#endif // TRITON_ASCEND_HAS_INPROC_COSTMODEL

void EstimateCVPipelineCostPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The CV pipeline already gave up on this module; there is nothing whose
  // cost would be meaningful.
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

#if TRITON_ASCEND_HAS_INPROC_COSTMODEL
  estimateModuleCost(module, hardwareConfigPath);
#else
  LOG_DEBUG("costmodel not built into this binary; skipping cost estimation");
#endif
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createEstimateCVPipelineCostPass(std::string hardwareConfigPath) {
  return std::make_unique<EstimateCVPipelineCostPass>(
      std::move(hardwareConfigPath));
}

void registerEstimateCVPipelineCostPasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createEstimateCVPipelineCostPass();
  });
}

} // namespace triton
} // namespace mlir
