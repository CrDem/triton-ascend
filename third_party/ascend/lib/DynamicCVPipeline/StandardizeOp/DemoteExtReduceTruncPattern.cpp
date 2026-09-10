#include "ascend/include/DynamicCVPipeline/StandardizeOp/PatternMatchRewrites.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir::triton::CVSplit {

DemoteExtReduceTruncPattern::DemoteExtReduceTruncPattern(MLIRContext *ctx,
                                                         bool enabled)
    : OpRewritePattern<arith::TruncFOp>(ctx, /*benefit=*/10), enabled(enabled) {}

LogicalResult
DemoteExtReduceTruncPattern::matchAndRewrite(arith::TruncFOp truncOp,
                                             PatternRewriter &rewriter) const {
  if (!enabled)
    return failure();

  Location loc = truncOp.getLoc();
  auto dstType = dyn_cast<RankedTensorType>(truncOp.getType());
  if (!dstType || !dstType.getElementType().isF16())
    return failure();

  // 1. Match producer: linalg.reduce
  auto reduceOp = truncOp.getIn().getDefiningOp<linalg::ReduceOp>();
  if (!reduceOp || !reduceOp->hasOneUse())
    return failure();

  if (reduceOp.getInputs().size() != 1 || reduceOp.getInits().size() != 1)
    return failure();

  Value reduceIn = reduceOp.getInputs().front();
  Value reduceInit = reduceOp.getInits().front();

  // 2. Match arith.extf feeding reduce input
  auto extOp = reduceIn.getDefiningOp<arith::ExtFOp>();
  if (!extOp)
    return failure();

  Value origF16Input = extOp.getIn();
  auto srcType = dyn_cast<RankedTensorType>(origF16Input.getType());
  if (!srcType || !srcType.getElementType().isF16())
    return failure();

  // 3. Inspect reduction body and combiner
  Block &body = reduceOp.getRegion().front();
  if (body.getNumArguments() != 2)
    return failure();

  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return failure();

  Operation *combOp = yieldOp.getValues().front().getDefiningOp();
  if (!combOp)
    return failure();

  if (combOp->getOperand(0) != body.getArgument(0) ||
      combOp->getOperand(1) != body.getArgument(1)) {
    return failure();
  }

  bool isSupported =
      isa<arith::MaximumFOp, arith::MinimumFOp, arith::AddFOp>(combOp);
  if (!isSupported)
    return failure();

  // 4. Trace the init tensor (fill <- empty)
  auto fillOp = reduceInit.getDefiningOp<linalg::FillOp>();
  if (!fillOp)
    return failure();

  Value initVal = fillOp.getInputs().front();
  auto cstOp = initVal.getDefiningOp<arith::ConstantFloatOp>();
  if (!cstOp)
    return failure();

  // 5. Create new f16 empty and fill
  auto initType = cast<RankedTensorType>(reduceInit.getType());
  auto f16InitType =
      initType.cloneWith(initType.getShape(), rewriter.getF16Type());

  SmallVector<Value> dynDims;
  for (int64_t i = 0; i < initType.getRank(); ++i) {
    if (initType.isDynamicDim(i)) {
      dynDims.push_back(rewriter.create<tensor::DimOp>(loc, reduceInit, i));
    }
  }
  Value newEmpty = rewriter.create<tensor::EmptyOp>(loc, f16InitType, dynDims);

  APFloat f16Val(cstOp.value().convertToDouble());
  bool losesInfo = false;
  f16Val.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &losesInfo);
  Value newCst = rewriter.create<arith::ConstantFloatOp>(loc, rewriter.getF16Type(), f16Val);
  Value newFill =
      rewriter.create<linalg::FillOp>(loc, newCst, newEmpty).getResult(0);

  // 6. Create f16 linalg.reduce matching the original combiner
  auto newReduce = rewriter.create<linalg::ReduceOp>(
      loc, ValueRange{origF16Input}, ValueRange{newFill},
      reduceOp.getDimensions(),
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value inF16 = args[0];
        Value initF16 = args[1];
        Value result;

        llvm::TypeSwitch<Operation *>(combOp)
            .Case<arith::MaximumFOp>([&](auto) {
              result = b.create<arith::MaximumFOp>(nestedLoc, inF16, initF16);
            })
            .Case<arith::MinimumFOp>([&](auto) {
              result = b.create<arith::MinimumFOp>(nestedLoc, inF16, initF16);
            })
            .Case<arith::AddFOp>([&](auto) {
              result = b.create<arith::AddFOp>(nestedLoc, inF16, initF16);
            });

        b.create<linalg::YieldOp>(nestedLoc, result);
      });

  // 7. Replace truncOp uses directly with f16 reduce result
  rewriter.replaceOp(truncOp, newReduce.getResults().front());
  return success();
}

} // namespace mlir::triton::CVSplit
