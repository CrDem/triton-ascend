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
// A loop whose bound is not a compile-time constant has no trip count, and its
// body is usually where the kernel spends its time. Such a bound comes either
// from a kernel argument or from the launch grid -- TritonToLinalg appends
// program_id and num_programs to the entry function's arguments, so naming
// them binds them. Failing that, say how many iterations to assume:
//
//   TRITON_ASCEND_CV_COST_ARG_BINDINGS=pid_x=0,num_programs_x=28,arg3=98432
//   TRITON_ASCEND_CV_COST_DEFAULT_TRIP_COUNT=32
//
// The hardware profile defaults to a 910B. Selecting another one matters on
// 910_95, whose data paths differ: the Cube accumulator can drain straight
// into UB, and UB and L1 exchange data on chip instead of through HBM. Which
// engine a transfer occupies is derived from the address spaces the compiler
// put on the memrefs, so those paths are costed as soon as the profile
// describes them:
//
//   TRITON_ASCEND_CV_COST_HARDWARE_CONFIG=/path/to/ascend_910_95.json
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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
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

/// Values for entry-function arguments, so that loops bounded by a kernel
/// argument can be resolved: "arg3=98432,arg5=128".
constexpr const char *kArgBindingsEnvVar = "TRITON_ASCEND_CV_COST_ARG_BINDINGS";

/// Iterations assumed for loops that no binding resolves. Defaults to 1, which
/// counts such a body exactly once.
constexpr const char *kDefaultTripCountEnvVar =
    "TRITON_ASCEND_CV_COST_DEFAULT_TRIP_COUNT";

/// Path to the hardware profile JSON. Without this there is no way to select
/// anything other than the built-in default, which is a 910B.
constexpr const char *kHardwareConfigEnvVar =
    "TRITON_ASCEND_CV_COST_HARDWARE_CONFIG";

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
  // Shape metadata: reshape/expand/collapse/cast rewrite how a buffer is
  // indexed, they never move an element. Charging them as vector work is a
  // pure over-estimate. (The memref equivalents are ViewLikeOpInterface and
  // are already covered above.)
  if (isa<tensor::ReshapeOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp,
          tensor::CastOp>(op)) {
    return true;
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

/// Cross-core synchronisation. These occupy real time -- a core waiting on a
/// flag is idle -- but modelling that needs the flag graph the CV pipeline
/// builds, which this pass does not read yet. Recognised explicitly so they
/// are reported as un-modelled rather than silently mistaken for compute.
bool isSyncOp(Operation *op) {
  return isa<hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp, hivm::SyncBlockOp,
             hivm::SyncBlockLockOp, hivm::SyncBlockUnlockOp,
             hivm::CreateSyncBlockLockOp>(op);
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

/// Costmodel name of the memory a buffer lives in, read from the address space
/// the compiler attached to its memref. Empty when there is none to read -- a
/// tensor that has not been bufferized yet, typically.
///
/// Reading this rather than inferring it from the operation's role is what
/// makes 910_95 expressible at all: there, the same memref.copy may target HBM
/// or UB or L1, and only the type says which.
llvm::StringRef getMemorySpaceName(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType) {
    return {};
  }
  auto addressSpace =
      dyn_cast_or_null<hivm::AddressSpaceAttr>(memrefType.getMemorySpace());
  if (!addressSpace) {
    return {};
  }
  switch (addressSpace.getAddressSpace()) {
  case hivm::AddressSpace::GM:
    return "hbm"; // the bandwidth tables call global memory "hbm"
  case hivm::AddressSpace::L1:
    return "l1";
  case hivm::AddressSpace::L0A:
    return "l0a";
  case hivm::AddressSpace::L0B:
    return "l0b";
  case hivm::AddressSpace::L0C:
    return "l0c";
  case hivm::AddressSpace::UB:
    return "ub";
  default:
    return {};
  }
}

/// Which data mover executes a transfer between two memory spaces.
///
/// The engine follows from where the data comes from and goes to, not from
/// which core issued it. On 910_95 that distinction matters: draining the Cube
/// accumulator to UB (FixPipeUB) and draining it to HBM (FixPipe) are different
/// engines, and UB<->L1 traffic never leaves the chip at all.
HWUnit getTransferUnit(llvm::StringRef src, llvm::StringRef dst) {
  if (src == "l0c") {
    return dst == "ub" ? HWUnit::FixPipeUB : HWUnit::FixPipe;
  }
  if (dst == "l1" || dst == "l0a" || dst == "l0b") {
    // Feeding the Cube: from HBM/L2 this is its MTE2; from UB (910_95) it is
    // the vector store engine writing on-chip instead of out to HBM.
    return src == "ub" ? HWUnit::MTE3 : HWUnit::CubeMTE2;
  }
  if (dst == "ub") {
    return HWUnit::VecMTE2; // from HBM/L2, or from L1 on 910_95
  }
  if (src == "ub") {
    return HWUnit::MTE3;
  }
  return HWUnit::VecMTE2;
}

/// Memory spaces to assume when the IR does not carry an address space, e.g.
/// while values are still tensors. Reproduces the 910B-shaped guess the pass
/// used before address spaces were read.
std::pair<llvm::StringRef, llvm::StringRef> guessTransferSpaces(bool isCube,
                                                                bool isLoad) {
  using Spaces = std::pair<llvm::StringRef, llvm::StringRef>;
  if (isCube) {
    return isLoad ? Spaces{"hbm", "l1"} : Spaces{"l0c", "hbm"};
  }
  return isLoad ? Spaces{"hbm", "ub"} : Spaces{"ub", "hbm"};
}

/// Source and destination spaces of a transfer, preferring what the IR says
/// and falling back to the guess above.
std::pair<llvm::StringRef, llvm::StringRef>
resolveTransferSpaces(Type srcType, Type dstType, bool isCube, bool isLoad) {
  llvm::StringRef src = getMemorySpaceName(srcType);
  llvm::StringRef dst = getMemorySpaceName(dstType);
  if (src.empty() || dst.empty()) {
    auto guessed = guessTransferSpaces(isCube, isLoad);
    if (src.empty()) {
      src = guessed.first;
    }
    if (dst.empty()) {
      dst = guessed.second;
    }
  }
  return {src, dst};
}

/// Both ends of an hivm data-movement operation.
///
/// hivm.hir.fixpipe and hivm.hir.copy are both built as (source, destination)
/// -- see InterCoreTransferAndSync, which creates them and labels the operands.
/// The roles come from that convention rather than from scanning for whichever
/// operand happens to carry an address space: fixpipe's source is often still a
/// tensor, so such a scan finds only the destination and, taking it for the
/// source, turns a Cube drain into a vector store.
struct TransferEnds {
  Value source;
  Value dest;
};

std::optional<TransferEnds> getHivmTransferEnds(Operation *op) {
  if (op->getNumOperands() < 2) {
    return std::nullopt;
  }
  return TransferEnds{op->getOperand(0), op->getOperand(1)};
}

/// Spaces of an hivm transfer, per side, falling back independently so that an
/// unreadable end never shifts the other end's role.
std::pair<llvm::StringRef, llvm::StringRef>
resolveHivmTransferSpaces(Operation *op, llvm::StringRef fallbackSrc,
                          llvm::StringRef fallbackDst) {
  llvm::StringRef src;
  llvm::StringRef dst;
  if (auto ends = getHivmTransferEnds(op)) {
    src = getMemorySpaceName(ends->source.getType());
    dst = getMemorySpaceName(ends->dest.getType());
  }
  return {src.empty() ? fallbackSrc : src, dst.empty() ? fallbackDst : dst};
}

/// Size of a transfer: what lands in the destination, falling back to whatever
/// shaped value the operation exposes.
Type getTransferSizingType(Operation *op) {
  if (auto ends = getHivmTransferEnds(op)) {
    if (isa<ShapedType>(ends->dest.getType())) {
      return ends->dest.getType();
    }
  }
  return getRepresentativeShapedType(op);
}

int64_t getTransferStartupLatency(HWUnit unit, const HardwareConfig &config) {
  switch (unit) {
  case HWUnit::CubeMTE2:
  case HWUnit::VecMTE2:
    return config.getMTE2StartupLatency();
  case HWUnit::FixPipe:
  case HWUnit::FixPipeUB:
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
  /// Recognised as something that does occupy hardware, but deliberately
  /// charged zero because no model exists yet. Cross-core synchronisation is
  /// the important case: it is real time that this pass cannot yet account
  /// for. Distinct from Generic (which does produce a number) and from the
  /// zero-cost ops (which really are free).
  NotModelled,
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
  case CostConfidence::NotModelled:
    return "not-modelled";
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

/// Cost of a bulk data transfer between two memory spaces.
OpCost estimateTransfer(Type shapedType, llvm::StringRef srcSpace,
                        llvm::StringRef dstSpace,
                        const HardwareConfig &config) {
  OpCost cost;
  cost.unit = getTransferUnit(srcSpace, dstSpace);

  auto bytes = getShapedByteSize(shapedType);
  if (!bytes) {
    cost.confidence = CostConfidence::UnknownSize;
    return cost;
  }
  cost.bytes = *bytes;

  int64_t transferCycles = config.estimateTransferCycles(
      srcSpace, dstSpace, *bytes, config.getActiveBandwidthCores());
  cost.cycles = transferCycles + getTransferStartupLatency(cost.unit, config);
  return cost;
}

/// Map an operation to the vector instruction whose measured cost it is closest
/// to. The names are tilesim mnemonics; the migrated cycle table distinguishes
/// cheap ALU ops from transcendentals, so this mapping is what makes an exp or
/// a divide cost more than an add. Returning an empty name means "no mapping",
/// which the caller reports as a gap rather than hiding.
llvm::StringRef getVectorIntrinsic(Operation *op) {
  using Ret = llvm::StringRef;
  return llvm::TypeSwitch<Operation *, Ret>(op)
      // Arithmetic.
      .Case<arith::AddFOp, arith::AddIOp>(
          [](Operation *) { return Ret("VADD"); })
      .Case<arith::SubFOp, arith::SubIOp>(
          [](Operation *) { return Ret("VSUB"); })
      .Case<arith::MulFOp, arith::MulIOp>(
          [](Operation *) { return Ret("VMUL"); })
      .Case<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(
          [](Operation *) { return Ret("VDIV"); })
      .Case<arith::MaxNumFOp, arith::MaximumFOp, arith::MaxSIOp,
            arith::MaxUIOp>([](Operation *) { return Ret("VMAX"); })
      .Case<arith::MinNumFOp, arith::MinimumFOp, arith::MinSIOp,
            arith::MinUIOp>([](Operation *) { return Ret("VMIN"); })
      .Case<arith::NegFOp>([](Operation *) { return Ret("VSUB"); })
      .Case<arith::SelectOp>([](Operation *) { return Ret("VSEL"); })
      .Case<arith::CmpFOp, arith::CmpIOp>(
          [](Operation *) { return Ret("VCMPV_GE"); })
      // Conversions all cost about one pass over the data.
      .Case<arith::TruncFOp, arith::ExtFOp, arith::TruncIOp, arith::ExtSIOp,
            arith::ExtUIOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
            arith::FPToUIOp, arith::BitcastOp>(
          [](Operation *) { return Ret("VCOPY"); })
      // Transcendentals.
      .Case<math::ExpOp, math::Exp2Op>([](Operation *) { return Ret("VEXP"); })
      .Case<math::LogOp, math::Log2Op>([](Operation *) { return Ret("LOG"); })
      .Case<math::SqrtOp, math::RsqrtOp>(
          [](Operation *) { return Ret("VSQRT"); })
      .Case<math::AbsFOp, math::AbsIOp>([](Operation *) { return Ret("VABS"); })
      // Structured ops that are really one pass over the data.
      .Case<linalg::FillOp, linalg::TransposeOp, linalg::CopyOp>(
          [](Operation *) { return Ret("VCOPY"); })
      .Case<linalg::BroadcastOp>([](Operation *) { return Ret("VBRCB"); })
      .Default([](Operation *) { return Ret(); });
}

/// Cost of a reduction. A reduction is not one pass over the data: after the
/// elementwise pass it costs a logarithmic tree inside each vector register and
/// another one across registers. Mirrors the costmodel's own reduce model.
OpCost estimateReduce(Operation *op, Type shapedType, int64_t elements,
                      const HardwareConfig &config) {
  OpCost cost;
  cost.unit = HWUnit::Vector;

  int elementBits = getShapedElementBits(shapedType);
  int64_t vectorWidth = elementBits > 0 ? 2048 / elementBits : 64;
  if (vectorWidth <= 0) {
    vectorWidth = 1;
  }
  int64_t numVectors = (elements + vectorWidth - 1) / vectorWidth;

  auto log2Steps = [](int64_t value) {
    int steps = 0;
    while (value > 1) {
      value /= 2;
      ++steps;
    }
    return steps;
  };

  cost.cycles = numVectors + log2Steps(vectorWidth) + log2Steps(numVectors) +
                config.getVectorStartupLatency();
  cost.flops = elements;
  return cost;
}

/// Cost of an elementwise compute op on the Vector core.
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

  if (isa<linalg::ReduceOp>(op)) {
    return estimateReduce(op, shapedType, *elements, config);
  }

  llvm::StringRef intrinsic = getVectorIntrinsic(op);
  cost.cycles = config.estimateVectorCyclesFromTable(
      *elements, getShapedElementBits(shapedType), intrinsic);
  cost.flops = *elements;
  // With a known instruction the cost comes from the measured table; without
  // one it falls back to a flat cycle per pass, which is a guess worth
  // surfacing rather than a model.
  cost.confidence = intrinsic.empty() ? CostConfidence::Generic
                                      : CostConfidence::Modelled;
  return cost;
}

/// Classify an operation and estimate what it costs. Returns nullopt for ops
/// that do not occupy any hardware pipe.
std::optional<OpCost> estimateOpCost(Operation *op,
                                     const HardwareConfig &config) {
  if (isZeroCostOp(op)) {
    return std::nullopt;
  }

  // Cross-core synchronisation: real time, no model yet.
  if (isSyncOp(op)) {
    OpCost cost;
    cost.unit = HWUnit::Scalar;
    cost.confidence = CostConfidence::NotModelled;
    return cost;
  }

  // Pointer arithmetic and scalar loads/stores emitted by the LLVM lowering.
  // They run on the scalar unit, outside the Cube/Vector roofline, so they are
  // charged zero -- but recognised, so they do not masquerade as compute of
  // unknown size.
  if (isa<LLVM::LLVMDialect>(op->getDialect())) {
    OpCost cost;
    cost.unit = HWUnit::Scalar;
    cost.confidence = CostConfidence::NotModelled;
    return cost;
  }

  const bool isCube = runsOnCubeCore(op);

  if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
    return estimateMatmul(matmulOp, config);
  }

  // hivm.hir.fixpipe drains the Cube accumulator. On 910_95 it may land in UB
  // instead of HBM, which is a different engine entirely, so the destination
  // is read from the operands rather than assumed.
  if (isa<hivm::FixpipeOp>(op)) {
    // The source is the accumulator by definition; only the destination is in
    // question, and on 910_95 it decides whether this is a FixPipe to HBM or
    // the on-chip drain into UB.
    auto [srcSpace, dstSpace] =
        resolveHivmTransferSpaces(op, /*fallbackSrc=*/"l0c",
                                  /*fallbackDst=*/"hbm");
    Type shapedType = getTransferSizingType(op);
    if (!shapedType) {
      OpCost cost;
      cost.unit = getTransferUnit(srcSpace, dstSpace);
      cost.confidence = CostConfidence::UnknownSize;
      return cost;
    }
    return estimateTransfer(shapedType, srcSpace, dstSpace, config);
  }

  if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
    // A copy into a local allocation is a load; a copy out of one is a
    // write-back. Used only when the memrefs carry no address space.
    const bool targetIsLocal = isBackedByLocalAlloc(copyOp.getTarget());
    const bool sourceIsLocal = isBackedByLocalAlloc(copyOp.getSource());
    const bool isLoad = targetIsLocal || !sourceIsLocal;
    auto [srcSpace, dstSpace] =
        resolveTransferSpaces(copyOp.getSource().getType(),
                              copyOp.getTarget().getType(), isCube, isLoad);
    return estimateTransfer(copyOp.getTarget().getType(), srcSpace, dstSpace,
                            config);
  }

  // hivm.hir.copy moves a buffer, it does not compute over it: it belongs to a
  // data mover, not to the vector ALU.
  if (isa<hivm::CopyOp>(op)) {
    // The CV pipeline emits these to stage data between the cores; on 910_95
    // the destination is often L1, keeping the traffic on chip.
    auto guessed = guessTransferSpaces(isCube, /*isLoad=*/true);
    auto [srcSpace, dstSpace] =
        resolveHivmTransferSpaces(op, guessed.first, guessed.second);
    Type shapedType = getTransferSizingType(op);
    if (!shapedType) {
      OpCost cost;
      cost.unit = getTransferUnit(srcSpace, dstSpace);
      cost.confidence = CostConfidence::UnknownSize;
      return cost;
    }
    return estimateTransfer(shapedType, srcSpace, dstSpace, config);
  }

  if (CVPipeline::isStoreLike(op)) {
    Type shapedType = getRepresentativeShapedType(op);
    auto [srcSpace, dstSpace] = guessTransferSpaces(isCube, /*isLoad=*/false);
    if (!shapedType) {
      OpCost cost;
      cost.unit = getTransferUnit(srcSpace, dstSpace);
      cost.confidence = CostConfidence::UnknownSize;
      return cost;
    }
    return estimateTransfer(shapedType, srcSpace, dstSpace, config);
  }

  // Anything left that touches no shaped value is scalar bookkeeping: index
  // queries, block ids, predicates. Checked after the specific kinds above so
  // it can never swallow one of them.
  if (!getRepresentativeShapedType(op)) {
    OpCost cost;
    cost.unit = HWUnit::Scalar;
    cost.confidence = CostConfidence::NotModelled;
    return cost;
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

/// What the caller can supply that the IR itself does not say.
///
/// A loop bounded by a kernel argument -- a sequence length, say -- has no
/// trip count at compile time, yet its body is usually where the kernel spends
/// its time. Counting such a body once understates the estimate by however many
/// iterations run, so there are two ways to fill the gap: bind the arguments to
/// the sizes the kernel will actually be called with, or, failing that, assume
/// a fixed iteration count.
struct TripCountOptions {
  /// Values for entry-function arguments, by argument index.
  llvm::DenseMap<unsigned, int64_t> argBindings;
  /// Values for the launch grid, keyed "pid_x" / "num_programs_x" and so on.
  llvm::StringMap<int64_t> gridBindings;
  /// Iteration count assumed for loops that remain unresolved. One reproduces
  /// the behaviour of not assuming anything.
  int64_t defaultTripCount = 1;
};

/// TritonToLinalg does not lower tl.program_id to an operation: it appends the
/// launch grid to the entry function's arguments and replaces the op with one
/// of them. The last three are program_id x/y/z and the three before them are
/// num_programs x/y/z -- see FunctionConverter.cpp, which does the append.
///
/// That makes the grid bindable through the ordinary argument mechanism, but
/// only if the caller counts arguments by hand. Naming the slots instead is
/// what makes `pid_x=0` work.
constexpr unsigned kLaunchGridRank = 3;

/// Grid slot a function argument corresponds to, or empty if it is an ordinary
/// argument. `numArgs` is the entry function's argument count.
std::string getGridBindingName(unsigned argIndex, unsigned numArgs) {
  if (numArgs < 2 * kLaunchGridRank) {
    return {}; // no launch grid was appended to this function
  }
  static constexpr llvm::StringLiteral kDims[] = {"x", "y", "z"};

  unsigned programIdBase = numArgs - kLaunchGridRank;
  if (argIndex >= programIdBase) {
    return ("pid_" + kDims[argIndex - programIdBase]).str();
  }
  unsigned numProgramsBase = numArgs - 2 * kLaunchGridRank;
  if (argIndex >= numProgramsBase) {
    return ("num_programs_" + kDims[argIndex - numProgramsBase]).str();
  }
  return {};
}

/// Resolve an integer value given bindings for the entry function's arguments.
///
/// Only entry-block arguments of a func.func are bound. This matters: a loop
/// induction variable and a loop-carried value are BlockArguments too, and
/// binding those by index would silently yield a plausible but wrong number.
std::optional<int64_t> evaluateWithBindings(Value value,
                                            const TripCountOptions &options,
                                            int depth = 0) {
  // Guards against a pathological or cyclic expression.
  if (depth > 16) {
    return std::nullopt;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto funcOp =
        dyn_cast_or_null<func::FuncOp>(blockArg.getOwner()->getParentOp());
    if (!funcOp || blockArg.getOwner() != &funcOp.getBody().front()) {
      return std::nullopt; // induction variable / iter_arg, not a kernel arg
    }
    unsigned argIndex = blockArg.getArgNumber();
    auto byIndex = options.argBindings.find(argIndex);
    if (byIndex != options.argBindings.end()) {
      return byIndex->second;
    }
    // Only consult the grid slots when the caller named some, so a kernel
    // whose real arguments happen to sit in those positions is unaffected.
    if (!options.gridBindings.empty()) {
      std::string gridName =
          getGridBindingName(argIndex, funcOp.getNumArguments());
      auto byName = options.gridBindings.find(gridName);
      if (!gridName.empty() && byName != options.gridBindings.end()) {
        return byName->second;
      }
    }
    return std::nullopt;
  }

  Operation *def = value.getDefiningOp();
  if (!def) {
    return std::nullopt;
  }

  if (auto constOp = dyn_cast<arith::ConstantOp>(def)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
    return std::nullopt;
  }

  auto operand = [&](unsigned index) {
    return evaluateWithBindings(def->getOperand(index), options, depth + 1);
  };

  // Pass-through casts.
  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
          arith::IndexCastOp>(def)) {
    return operand(0);
  }

  if (def->getNumOperands() < 2) {
    return std::nullopt;
  }
  auto lhs = operand(0);
  auto rhs = operand(1);
  if (!lhs || !rhs) {
    return std::nullopt;
  }

  if (isa<arith::AddIOp>(def)) {
    return *lhs + *rhs;
  }
  if (isa<arith::SubIOp>(def)) {
    return *lhs - *rhs;
  }
  if (isa<arith::MulIOp>(def)) {
    return *lhs * *rhs;
  }
  if (isa<arith::DivSIOp, arith::DivUIOp>(def)) {
    if (*rhs == 0) {
      return std::nullopt;
    }
    return *lhs / *rhs;
  }
  // Rounding divisions matter more than they look: AddControlFlowCondition
  // rewrites every pipelined loop's upper bound as
  //   lb + step * (ceildiv(ceildiv(ub - lb, step) * buffers, x) + ifCount)
  // so without these a loop with entirely constant bounds still looks dynamic.
  if (isa<arith::CeilDivUIOp>(def)) {
    if (*rhs == 0) {
      return std::nullopt;
    }
    auto unsignedLhs = static_cast<uint64_t>(*lhs);
    auto unsignedRhs = static_cast<uint64_t>(*rhs);
    return static_cast<int64_t>((unsignedLhs + unsignedRhs - 1) / unsignedRhs);
  }
  if (isa<arith::CeilDivSIOp>(def)) {
    if (*rhs == 0) {
      return std::nullopt;
    }
    int64_t quotient = *lhs / *rhs;
    int64_t remainder = *lhs % *rhs;
    // Truncation rounds towards zero; step back up when the exact result was
    // positive and inexact.
    if (remainder != 0 && ((remainder > 0) == (*rhs > 0))) {
      ++quotient;
    }
    return quotient;
  }
  if (isa<arith::FloorDivSIOp>(def)) {
    if (*rhs == 0) {
      return std::nullopt;
    }
    int64_t quotient = *lhs / *rhs;
    int64_t remainder = *lhs % *rhs;
    if (remainder != 0 && ((remainder < 0) != (*rhs < 0))) {
      --quotient;
    }
    return quotient;
  }
  if (isa<arith::RemSIOp, arith::RemUIOp>(def)) {
    if (*rhs == 0) {
      return std::nullopt;
    }
    return *lhs % *rhs;
  }
  if (isa<arith::MinSIOp, arith::MinUIOp>(def)) {
    return std::min(*lhs, *rhs);
  }
  if (isa<arith::MaxSIOp, arith::MaxUIOp>(def)) {
    return std::max(*lhs, *rhs);
  }
  return std::nullopt;
}

/// Trip count of a loop once the supplied bindings are taken into account.
std::optional<int64_t>
resolveTripCount(scf::ForOp forOp, const TripCountOptions &options) {
  // Runs even with no bindings: a bound can be entirely constant yet still
  // defeat the simpler static analysis, because the pipelining rewrite buries
  // it under rounding divisions.
  auto lower = evaluateWithBindings(forOp.getLowerBound(), options);
  auto upper = evaluateWithBindings(forOp.getUpperBound(), options);
  auto step = evaluateWithBindings(forOp.getStep(), options);
  if (!lower || !upper || !step || *step == 0) {
    return std::nullopt;
  }
  if (*upper <= *lower) {
    return 0;
  }
  return (*upper - *lower + *step - 1) / *step;
}

struct LoopWeight {
  int64_t multiplier = 1;
  /// Innermost enclosing loop whose trip count could not be determined, even
  /// with bindings. Null means `multiplier` is trustworthy. The whole body of
  /// such a loop is assumed together, so the loop -- not each operation in it
  /// -- is the thing worth reporting.
  Operation *dynamicLoop = nullptr;
  /// Iteration count assumed for `dynamicLoop`, so the report can say what
  /// the estimate is actually based on.
  int64_t assumedTripCount = 1;
};

/// How many times an operation executes, given the enclosing loop nest.
///
/// Each loop is resolved in three steps, most trustworthy first: constant
/// bounds in the IR, then the caller's argument bindings, then the assumed
/// default. Only a loop that reaches the third step is reported, since only
/// then is the number a guess.
LoopWeight getLoopWeight(Operation *op, const TripCountOptions &options) {
  LoopWeight weight;
  // Walking outwards, so the first unresolved loop met is the innermost one.
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto forOp = dyn_cast<scf::ForOp>(parent);
    if (!forOp) {
      continue;
    }

    auto tripCount = mlir::ascend::utils::analyzeScfForTripCount(forOp);
    if (tripCount.isStatic) {
      weight.multiplier *= tripCount.staticTripCount;
      continue;
    }
    if (auto bound = resolveTripCount(forOp, options)) {
      weight.multiplier *= *bound;
      continue;
    }

    weight.multiplier *= options.defaultTripCount;
    if (!weight.dynamicLoop) {
      weight.dynamicLoop = forOp;
      weight.assumedTripCount = options.defaultTripCount;
    }
  }
  return weight;
}

/// Canonical name for a launch-grid binding. Accepts the spellings the
/// standalone costmodel uses, so the same string works in both. Empty when the
/// name is not a grid slot and should be read as an argument index.
std::string canonicalGridName(llvm::StringRef name) {
  auto isDim = [](llvm::StringRef dim) {
    return dim == "x" || dim == "y" || dim == "z";
  };
  if (name.consume_front("pid_") || name.consume_front("program_id_")) {
    return isDim(name) ? ("pid_" + name).str() : std::string();
  }
  if (name.consume_front("num_programs_")) {
    return isDim(name) ? ("num_programs_" + name).str() : std::string();
  }
  return {};
}

/// Read the trip-count options from the environment.
///
/// Bindings are "arg3=98432,arg5=128" (the "arg" prefix is optional), naming
/// entry-function argument indices. The environment is process-wide, which is
/// fine for inspecting one kernel but is the reason these are not the right
/// channel for per-configuration autotuning; that will want the values passed
/// in explicitly.
TripCountOptions readTripCountOptions() {
  TripCountOptions options;

  if (const char *raw = std::getenv(kArgBindingsEnvVar)) {
    llvm::SmallVector<llvm::StringRef> entries;
    llvm::StringRef(raw).split(entries, ',', /*MaxSplit=*/-1,
                               /*KeepEmpty=*/false);
    for (llvm::StringRef entry : entries) {
      auto [rawName, rawValue] = entry.split('=');
      llvm::StringRef name = rawName.trim();
      int64_t value = 0;
      if (rawValue.trim().getAsInteger(10, value)) {
        LOG_DEBUG("ignoring malformed binding: " << entry);
        continue;
      }
      if (std::string gridName = canonicalGridName(name); !gridName.empty()) {
        options.gridBindings[gridName] = value;
        continue;
      }
      llvm::StringRef indexText = name;
      indexText.consume_front("arg");
      unsigned index = 0;
      if (indexText.getAsInteger(10, index)) {
        LOG_DEBUG("ignoring malformed binding: " << entry);
        continue;
      }
      options.argBindings[index] = value;
    }
  }

  if (const char *raw = std::getenv(kDefaultTripCountEnvVar)) {
    int64_t value = 0;
    if (!llvm::StringRef(raw).trim().getAsInteger(10, value) && value > 0) {
      options.defaultTripCount = value;
    } else {
      LOG_DEBUG("ignoring invalid default trip count: " << raw);
    }
  }

  return options;
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

  // FixPipeUB is Cube-side work even though it lands in Vector memory: it is
  // the Cube's own drain engine. Leaving it out here would silently drop its
  // cycles from the estimate entirely.
  int64_t cubePathCycles = std::max({cyclesOf(HWUnit::Cube),
                                     cyclesOf(HWUnit::CubeMTE2),
                                     cyclesOf(HWUnit::FixPipe),
                                     cyclesOf(HWUnit::FixPipeUB)});

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
  case CostConfidence::NotModelled:
    return 2;
  case CostConfidence::UnknownSize:
    return 3;
  }
  return 0;
}

/// A loop whose trip count is unknown, summarised by what it contains. One
/// such loop understates the estimate once, not once per operation inside it,
/// so this is the granularity the report uses.
struct DynamicLoopStats {
  int64_t costedOps = 0;
  int64_t bodyCycles = 0;
  int64_t assumedTripCount = 1;
};

struct CostBreakdown {
  llvm::MapVector<llvm::StringRef, OpKindStats> byOpKind;
  llvm::SmallVector<Operation *> unknownSizeOps;
  llvm::MapVector<Operation *, DynamicLoopStats> dynamicLoops;
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
    // Attribute the operation to its loop rather than listing it separately.
    // Zero-cycle operations are skipped: they cannot understate anything.
    if (weight.dynamicLoop && cost.cycles > 0) {
      DynamicLoopStats &loopStats = dynamicLoops[weight.dynamicLoop];
      loopStats.costedOps += 1;
      loopStats.bodyCycles += cost.cycles * weight.multiplier;
      loopStats.assumedTripCount = weight.assumedTripCount;
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
  auto listKinds = [&](CostConfidence confidence, llvm::StringRef what) {
    llvm::SmallVector<llvm::StringRef> kinds;
    for (const auto &[name, stats] : rows) {
      if (stats.confidence == confidence) {
        kinds.push_back(name);
      }
    }
    if (kinds.empty()) {
      return;
    }
    os << "[" << DEBUG_TYPE << "] " << kinds.size() << " operation kind(s) "
       << what << ":\n";
    for (llvm::StringRef name : kinds) {
      os << "[" << DEBUG_TYPE << "]     " << name << "\n";
    }
  };

  listKinds(CostConfidence::Generic,
            "charged by element count only, i.e. with no dedicated cost model");
  listKinds(CostConfidence::NotModelled,
            "recognised but charged zero: they occupy hardware time this pass "
            "does not account for yet (synchronisation, scalar work)");

  printOpList(os, "operation(s) with a non-static shape, contributing 0 cycles",
              breakdown.unknownSizeOps);

  // Loops, not their contents: one loop with an unknown trip count is one
  // problem, however many operations it happens to contain.
  if (!breakdown.dynamicLoops.empty()) {
    os << "[" << DEBUG_TYPE << "] " << breakdown.dynamicLoops.size()
       << " loop(s) whose trip count no binding resolved; the count below is"
          " assumed, so the estimate is only as good as that assumption. Set "
       << kArgBindingsEnvVar << " to bind the kernel arguments that bound them,"
          " or "
       << kDefaultTripCountEnvVar << " to change the assumption:\n";
    for (const auto &[loop, stats] : breakdown.dynamicLoops) {
      os << "[" << DEBUG_TYPE << "]     " << stats.costedOps
         << " costed op(s), " << stats.bodyCycles << " cycles over an assumed "
         << stats.assumedTripCount << " iteration(s)  at " << loop->getLoc()
         << "\n";
    }
  }
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
  // An explicitly constructed path wins; otherwise the environment selects the
  // profile, so that a 910_95 run does not silently get 910B numbers.
  std::string configPath = hardwareConfigPath.str();
  if (configPath.empty()) {
    if (const char *fromEnv = std::getenv(kHardwareConfigEnvVar)) {
      configPath = fromEnv;
    }
  }

  std::string configError;
  auto config =
      mlir::ascend::loadHardwareConfigForAnalysis(configPath, configError);
  if (!config) {
    LOG_DEBUG("failed to load hardware config: " << configError);
    return;
  }

  const TripCountOptions tripCountOptions = readTripCountOptions();

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

    LoopWeight weight = getLoopWeight(op, tripCountOptions);
    // Counts operations whose own cost could not be computed. An operation in
    // a loop of unknown trip count is not one of those: its per-iteration cost
    // is known, it is the iteration count that is not, which is accounted for
    // per loop rather than per operation.
    if (cost->confidence == CostConfidence::UnknownSize) {
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
  module->setAttr(
      kCVPipelineCostDynamicLoops,
      builder.getI64IntegerAttr(
          static_cast<int64_t>(breakdown.dynamicLoops.size())));

  auto reportTo = [&](llvm::raw_ostream &os) {
    os << "[" << DEBUG_TYPE << "] " << config->getName() << ": " << totalCycles
       << " cycles roofline ("
       << llvm::format("%.3f", config->cyclesToMicroseconds(totalCycles))
       << " us), " << scheduledCycles << " cycles critical path, " << nextOpId
       << " ops costed, " << unknownOps << " of unknown cost, "
       << breakdown.genericOps << " without a dedicated model, "
       << breakdown.dynamicLoops.size() << " loop(s) of unknown trip count\n";
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
