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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMPUTE_BLOCK_ID_MANAGER_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMPUTE_BLOCK_ID_MANAGER_H

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/Operation.h"

#include "DynamicCVPipeline/Common/Utils.h"

namespace mlir {
namespace CVPipeline {

/**
 * the class is to promise CUBEID and VECTORID are unified.
 */
class ComputeBlockIdManager {
public:
  ComputeBlockIdManager(Operation *root);
  bool isSameBlock(Operation *a, Operation *b);
  bool isWholeCubeReady(Operation *seedOp,
                        llvm::DenseMap<Operation *, int> &indegree);

  llvm::LogicalResult markOpBlockId(Operation *op);
  llvm::LogicalResult markOpsWithNewId(llvm::SmallVectorImpl<Operation *> &ops);
  void updateBlockId(Operation *op, int blockId);

  /// Drop every record of `op`, to be called immediately BEFORE erasing it.
  ///
  /// This class caches raw Operation pointers in both directions and nothing
  /// else removes them, so an operation erased from the IR stays in the maps as
  /// a dangling pointer. That is not theoretical: getOpsByBlockId hands the
  /// vector straight to callers who dereference it -- cloneDepSubgraph in
  /// RefineArgsBlockId walks it calling getBlock() and isBeforeInBlock(), and
  /// willCreateCycle seeds its "ok" set from it -- which segfaults on a freed
  /// operation. The manager also outlives a single loop, so an erase in one
  /// main loop crashes the next one.
  ///
  /// Removing the key from opToBlockId matters as much as removing it from the
  /// vector: MLIR allocates operations from a pool, so a later operation can
  /// land on the same address and silently inherit the dead one's block id.
  void forgetOp(Operation *op);

  bool shouldInheritFromParent(Block *block, CoreType requiredCoreType) const;
  llvm::LogicalResult inheritFromParent(Block *block);

  llvm::SmallVector<Operation *> getOpsByBlockId(int blockId) const;
  llvm::ArrayRef<Operation *> getOpsRefByBlockId(int blockId) const;

  // Get operations that share the same block_id AND mlir block of op
  llvm::SmallVector<Operation *> getOpsInSameBlock(Operation *op) const;

  std::optional<int> getBlockIdByOpOpt(Operation *op) const;
  int getNextId();

  int getBlockIdByOp(Operation *op);

  ~ComputeBlockIdManager() = default;
  ComputeBlockIdManager(const ComputeBlockIdManager &) = delete;
  ComputeBlockIdManager &operator=(const ComputeBlockIdManager &) = delete;
  ComputeBlockIdManager(ComputeBlockIdManager &&) = delete;
  ComputeBlockIdManager &operator=(ComputeBlockIdManager &&) = delete;

private:
  int cntComputeBlockId = 0;
  llvm::DenseMap<int, llvm::SmallVector<Operation *>> blockIdToOps;
  llvm::DenseMap<Operation *, int> opToBlockId;
  static constexpr int kBlockIdWidth = 32;
  llvm::LogicalResult markAndRecord(Operation *op, int blockId);
};

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMPUTE_BLOCK_ID_MANAGER_H
