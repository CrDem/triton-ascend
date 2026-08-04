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

#include <string_view>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.h"

#include "DynamicCVPipeline/Common/Utils.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "TritonToUnstructure/OffsetAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <fstream>
#include <string>
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
static constexpr const char *DEBUG_TYPE = "ReorderOpsByBlockIdPass";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace triton;
using namespace CVPipeline;

namespace {

// A dependency DAG of both SSA and memory of the ops
struct BlockOpGraph {
  Block *block;
  ArrayRef<Operation *> ops;
  DenseMap<Operation *, unsigned> opIndex;               // op → position in ops
  DenseMap<Operation *, SmallVector<Operation *>> preds; // op → its defs
  DenseMap<Operation *, SmallVector<Operation *>> succs; // op → its uses
  BlockOpGraph(ArrayRef<Operation *> allOps, Block *block,
               const MemoryDependenceGraph &memGraph);
};

void dumpBlockOpGraphToDot(const BlockOpGraph &graph, const std::string &filename) {
    std::ofstream os(filename);
    if (!os.is_open()) {
        llvm::errs() << "Failed to open file for DOT dump: " << filename << "\n";
        return;
    }

    os << "digraph BlockOpGraph {\n";
    os << "  rankdir=TB;\n"; // Top-to-bottom layout
    os << "  node [shape=box, fontname=\"Courier\", style=\"filled\", fillcolor=\"#f0f4f8\"];\n";
    os << "  edge [color=\"#333333\", arrowhead=\"normal\"];\n\n";

    // 1. Emit all nodes with formatted MLIR op representations
    for (size_t i = 0; i < graph.ops.size(); ++i) {
        mlir::Operation *op = graph.ops[i];

        // Print operation to a string
        std::string opStr;
        llvm::raw_string_ostream ss(opStr);
        op->print(ss, mlir::OpPrintingFlags().skipRegions().printGenericOpForm());

        // Escape special characters for DOT syntax
        std::string safeLabel;
        for (char c : ss.str()) {
            if (c == '"') safeLabel += "\\\"";
            else if (c == '\\') safeLabel += "\\\\";
            else if (c == '\n') safeLabel += "\\l"; // Left-align line breaks in Graphviz
            else safeLabel += c;
        }

        os << "  Node_" << i << " [label=\"[" << i << "] " 
           << op->getName().getStringRef().str() << "\\l" 
           << safeLabel << "\"];\n";
    }

    os << "\n  // Edges (Successors)\n";

    // 2. Emit directed edges from 'succs' map
    for (const auto &[op, successors] : graph.succs) {
        auto srcIt = graph.opIndex.find(op);
        if (srcIt == graph.opIndex.end()) continue;
        unsigned srcIdx = srcIt->second;

        for (mlir::Operation *succOp : successors) {
            auto dstIt = graph.opIndex.find(succOp);
            if (dstIt != graph.opIndex.end()) {
                unsigned dstIdx = dstIt->second;
                os << "  Node_" << srcIdx << " -> Node_" << dstIdx << ";\n";
            }
        }
    }

    os << "}\n";
    os.close();
    llvm::errs() << "Successfully dumped BlockOpGraph to " << filename << "\n";
}

// Helper class to manage edges in OpGraph, mainly to reduce congitive
// complexity of the build function
struct EdgeHelper {
  BlockOpGraph &graph;
  DenseSet<std::pair<Operation *, Operation *>> seen;
  Block *block;

  // find the ancestor directly in the block, and in opIndex; return nullptr if
  // either fails
  Operation *resolveToBlockOp(Operation *op);

  template <bool IsMemory = false>
  void addEdge(Operation *pred, Operation *succ);

  template <bool IsMemory = false>
  void addEdgeToUser(Operation *op, Operation *user) {
    if (graph.opIndex.contains(user)) {
      return; // same-level use, already covered by the def-side loop
    }
    Operation *ancestor = resolveToBlockOp(user);
    addEdge<IsMemory>(op, ancestor);
  };

  EdgeHelper(BlockOpGraph &g, Block *block) : graph(g), block(block) {};
};

} // namespace

Operation *EdgeHelper::resolveToBlockOp(Operation *op) {
  if (graph.opIndex.contains(op)) {
    return op;
  }
  Operation *ancestor = getAncestorInBlock(op, block);
  if (!ancestor || !graph.opIndex.contains(ancestor)) {
    return nullptr;
  }
  return ancestor;
}

template <bool IsMemory>
void EdgeHelper::addEdge(Operation *pred, Operation *succ) {
  if (!pred || !succ || pred == succ) {
    return;
  }
  if (seen.insert({pred, succ}).second) {
    LOG_DEBUG("Adding " << (IsMemory ? "memory " : "") << "edge from " << *pred
                        << " to " << *succ << "\n");
    graph.succs[pred].push_back(succ);
    graph.preds[succ].push_back(pred);
  }
};

BlockOpGraph::BlockOpGraph(ArrayRef<Operation *> allOps, Block *block,
                           const MemoryDependenceGraph &memGraph)
    : block(block), ops(allOps) {
  for (unsigned i = 0; i < allOps.size(); ++i) {
    opIndex[allOps[i]] = i;
    preds[allOps[i]]; // ensure every node has an entry
    succs[allOps[i]];
  }

  EdgeHelper edges(*this, block);

  for (Operation *op : allOps) {
    LOG_DEBUG("Processing op: " << *op << "\n");
    // Edges from operand defs (including defs nested inside other ops).
    for (Value const operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp) {
        continue;
      }
      Operation *def = edges.resolveToBlockOp(defOp);
      edges.addEdge(def, op);
    }

    // Edges from uses that live inside nested regions of another block-level
    // op.
    for (Value const result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        edges.addEdgeToUser(op, user);
      }
    }

    for (auto *memDef : memGraph.getExecBefore(op)) {
      Operation *def = edges.resolveToBlockOp(memDef);
      edges.addEdge<true>(def, op);
    }

    for (auto *memUser : memGraph.getExecAfter(op)) {
      edges.addEdgeToUser<true>(op, memUser);
    }
  }
}

static llvm::FailureOr<DenseMap<Operation *, int>>
collectBlockIds(ArrayRef<Operation *> allOps, ComputeBlockIdManager &bm) {
  DenseMap<Operation *, int> opBlockId;
  for (Operation *op : allOps) {
    if (llvm::failed(verifyOpBlockId(op))) {
      return llvm::failure();
    }
    auto blockIdOpt = getOpBlockId(op);
    if (blockIdOpt.has_value()) {
      opBlockId[op] = blockIdOpt.value();
      continue;
    }

    auto result = op->walk([&](Operation *nestedOp) {
      if (nestedOp != op &&
          !llvm::isa<scf::YieldOp, linalg::FillOp>(nestedOp)) {
        return WalkResult::interrupt();
      }
      auto currBlockIdOpt = getOpBlockId(nestedOp);
      if (!blockIdOpt.has_value()) {
        blockIdOpt = getOpBlockId(nestedOp);
      }
      if (currBlockIdOpt.has_value() && currBlockIdOpt != blockIdOpt) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted() || !blockIdOpt.has_value()) {
      blockIdOpt = bm.getNextId();
    } else {
      bm.updateBlockId(op, blockIdOpt.value());
    }
    opBlockId[op] = blockIdOpt.value();
  }
  return opBlockId;
}

namespace {

// Helper structure to hold the group-level graph data.
struct GroupAdjacencyGraph {
  Block *block;
  SmallVector<int> groupIds;
  SmallVector<SmallVector<unsigned>> succs;
  SmallVector<unsigned> inDeg;
  GroupAdjacencyGraph(const BlockOpGraph &g,
                      const DenseMap<Operation *, int> &opBlockId);
  llvm::FailureOr<SmallVector<int>> computeTopologicalOrder();
};

void dumpGroupAdjacencyGraphToDot(const GroupAdjacencyGraph &graph, const std::string &filename) {
    std::ofstream os(filename);
    if (!os.is_open()) {
        llvm::errs() << "Failed to open file for DOT dump: " << filename << "\n";
        return;
    }

    os << "digraph GroupAdjacencyGraph {\n";
    os << "  rankdir=TB;\n"; // Top-to-bottom layout
    
    // Aesthetic styling for group nodes
    os << "  node [shape=Mrecord, fontname=\"Helvetica\", style=\"filled\", fillcolor=\"#e2f0d9\", color=\"#548235\"];\n";
    os << "  edge [color=\"#385723\", penwidth=1.5];\n\n";

    // 1. Emit nodes (Group IDs)
    // The graph structure implies indices 0...groupIds.size()-1 map to the actual group IDs
    for (size_t i = 0; i < graph.groupIds.size(); ++i) {
        int actualGroupId = graph.groupIds[i];
        unsigned inDegree = (i < graph.inDeg.size()) ? graph.inDeg[i] : 0;
        
        // Node format: Index | Group ID | In-Degree
        os << "  Node_" << i << " [label=\"{Idx: " << i 
           << " | Group ID: " << actualGroupId 
           << " | In-Deg: " << inDegree << "}\"];\n";
    }

    os << "\n  // Edges (Group Successors)\n";

    // 2. Emit edges
    for (size_t i = 0; i < graph.succs.size(); ++i) {
        for (unsigned targetIdx : graph.succs[i]) {
            os << "  Node_" << i << " -> Node_" << targetIdx << ";\n";
        }
    }

    os << "}\n";
    os.close();
    llvm::errs() << "Successfully dumped GroupAdjacencyGraph to " << filename << "\n";
}

} // namespace

/**
 * Step 1: Build the group-level dependency graph from operator-level edges.
 * Maps individual operations to their respective groups and identifies
 * dependencies between those groups.
 */
GroupAdjacencyGraph::GroupAdjacencyGraph(
    const BlockOpGraph &g, const DenseMap<Operation *, int> &opBlockId)
    : block(g.block) {
  // 1. Collect distinct group IDs while preserving the first-appearance order.
  DenseSet<int> seenIds;
  for (Operation *op : g.ops) {
    int id = opBlockId.at(op);
    if (seenIds.insert(id).second) {
      groupIds.push_back(id);
    }
  }

  unsigned n = groupIds.size();
  succs.resize(n);
  inDeg.assign(n, 0);

  // Map group ID to its index in the groupIds vector for fast lookup.
  DenseMap<int, unsigned> groupPos;
  for (unsigned i = 0; i < n; ++i) {
    groupPos[groupIds[i]] = i;
  }

  // 2. Build group-level edges. Use a set to avoid duplicate edges between
  // groups.
  DenseSet<std::pair<unsigned, unsigned>> addedEdges;
  for (Operation *op : g.ops) {
    unsigned fromIdx = groupPos[opBlockId.at(op)];

    for (Operation *succ : g.succs.at(op)) {
      unsigned toIdx = groupPos[opBlockId.at(succ)];
      // Ignore intra-group dependencies and duplicate inter-group edges.
      if (fromIdx != toIdx && addedEdges.insert({fromIdx, toIdx}).second) {
        succs[fromIdx].push_back(toIdx);
        inDeg[toIdx]++;
      }
    }
  }

  // Logging the constructed group graph.
  LOG_DEBUG("Group-level edges:\n");
  for (unsigned i = 0; i < n; ++i) {
    LOG_DEBUG("  Group " << groupIds[i] << " -> ");
    for (unsigned succIdx : succs[i]) {
      LOG_DEBUG(groupIds[succIdx] << " ");
    }
    LOG_DEBUG("\n");
  }
}

/**
 * Step 2: Perform a topological sort (Kahn's Algorithm) on the group graph.
 * Returns the group IDs in an order that satisfies all dependencies.
 */
llvm::FailureOr<SmallVector<int>>
GroupAdjacencyGraph::computeTopologicalOrder() {
  SmallVector<int> result;
  SmallVector<unsigned> ready; // Nodes with in-degree 0.
  unsigned n = groupIds.size();

  for (unsigned i = 0; i < n; ++i) {
    if (inDeg[i] == 0) {
      ready.push_back(i);
    }
  }

  while (!ready.empty()) {
    auto cur = ready.pop_back_val();

    result.push_back(groupIds[cur]);

    for (unsigned succIdx : succs[cur]) {
      if (--inDeg[succIdx] == 0) {
        ready.push_back(succIdx);
      }
    }
  }

  LOG_DEBUG("Group order: ");
  for (int id : result) {
    LOG_DEBUG(id << " ");
  }
  LOG_DEBUG("\n");

  if (result.size() == n) {
    return result;
  }
  Operation *op = block->getParentOp();
  constexpr std::string_view kErrorPrefix =
      "Failed to compute topological order for ";
  if (!op) {
    llvm::errs() << kErrorPrefix
                 << "an unknown block that is not contained in an op";
    return llvm::failure();
  }
  size_t regionIdx = 0;
  bool found = false;
  for (auto [i, region] : llvm::enumerate(op->getRegions())) {
    for (auto &possibleBlock : region.getBlocks()) {
      if (&possibleBlock == block) {
        regionIdx = i;
      }
    }
  }
  op->emitError(kErrorPrefix) << "block in region " << regionIdx;
  return llvm::failure();
}

// Stable sort ops based on their group orders
static llvm::FailureOr<SmallVector<Operation *>>
buildReorderedOps(const BlockOpGraph &graph,
                  const DenseMap<Operation *, int> &opBlockId) {
  SmallVector<Operation *> reordered;
  GroupAdjacencyGraph adjacencyGraph{graph, opBlockId};
  auto groupOrderResult = adjacencyGraph.computeTopologicalOrder();
  if (llvm::failed(groupOrderResult)) {
    return llvm::failure();
  }

  for (int const blockId : groupOrderResult.value()) {
    for (Operation *op : graph.ops) {
      if (opBlockId.at(op) == blockId) {
        reordered.push_back(op);
      }
    }
  }
  return reordered;
}

// Reorder the ops in the mlir representation
static void applyReorder(Block &block, ArrayRef<Operation *> reordered) {
  Operation *terminator =
      block.mightHaveTerminator() ? block.getTerminator() : nullptr;
  for (Operation *op : reordered) {
    op->moveBefore(&block, block.end());
  }

  if (terminator) {
    terminator->moveBefore(&block, block.end());
  }
}

static llvm::LogicalResult
reorderOpsInBlock(Block &block, const MemoryDependenceGraph &memGraph,
                  ComputeBlockIdManager &bm) {
  const auto allOps =
      llvm::to_vector(llvm::make_pointer_range(block.without_terminator()));

  const BlockOpGraph graph{allOps, &block, memGraph};
  llvm::FailureOr<DenseMap<Operation *, int>> opBlockIdOpt =
      collectBlockIds(allOps, bm);
  if (failed(opBlockIdOpt)) {
    return failure();
  }

  auto &opBlockId = *opBlockIdOpt;
  LOG_DEBUG("Initial opBlockIds:\n");
  for (Operation *op : allOps) {
    LOG_DEBUG("  Op: " << *op << ", opBlockId = " << opBlockId[op] << "\n");
  }

  const auto reorderedRes = buildReorderedOps(graph, opBlockId);
  if (failed(reorderedRes)) {
    return failure();
  }

  applyReorder(block, reorderedRes.value());

  return llvm::success();
}

void dumpMemoryDependenceGraphToDot(const MemoryDependenceGraph &graph, 
                                    ArrayRef<Operation *> ops, 
                                    const std::string &filename) {
    std::ofstream os(filename);
    if (!os.is_open()) {
        llvm::errs() << "Failed to open file for DOT dump: " << filename << "\n";
        return;
    }

    os << "digraph MemoryDependenceGraph {\n";
    os << "  rankdir=TB;\n"; // Top-to-bottom layout
    os << "  node [shape=box, fontname=\"Courier\", style=\"filled\", fillcolor=\"#f9f9f9\"];\n";
    os << "  // Solid Blue = Memory Data Dep | Dashed Red = Execution Order Dep\n\n";

    // 1. Map operations to zero-based indices for clean node IDs
    DenseMap<Operation *, unsigned> opIndex;
    for (size_t i = 0; i < ops.size(); ++i) {
        opIndex[ops[i]] = i;

        // Extract printed MLIR string representation
        std::string opStr;
        llvm::raw_string_ostream ss(opStr);
        ops[i]->print(ss, mlir::OpPrintingFlags().skipRegions().printGenericOpForm());

        // Escape string for Graphviz DOT syntax
        std::string safeLabel;
        for (char c : ss.str()) {
            if (c == '"') safeLabel += "\\\"";
            else if (c == '\\') safeLabel += "\\\\";
            else if (c == '\n') safeLabel += "\\l"; // Left-align line breaks
            else safeLabel += c;
        }

        os << "  Node_" << i << " [label=\"[" << i << "] " 
           << ops[i]->getName().getStringRef().str() << "\\l" 
           << safeLabel << "\"];\n";
    }

    os << "\n  // 1. Memory Dependencies (Solid Blue)\n";
    for (size_t i = 0; i < ops.size(); ++i) {
        Operation *srcOp = ops[i];
        ArrayRef<Operation *> users = graph.getMemUsers(srcOp);
        for (Operation *dstOp : users) {
            auto it = opIndex.find(dstOp);
            if (it != opIndex.end()) {
                os << "  Node_" << i << " -> Node_" << it->second 
                   << " [color=\"#1f77b4\", label=\"mem\", fontcolor=\"#1f77b4\"];\n";
            }
        }
    }

    os << "\n  // 2. Execution Order Dependencies (Dashed Red)\n";
    for (size_t i = 0; i < ops.size(); ++i) {
        Operation *srcOp = ops[i];
        ArrayRef<Operation *> afterOps = graph.getExecAfter(srcOp);
        for (Operation *dstOp : afterOps) {
            auto it = opIndex.find(dstOp);
            if (it != opIndex.end()) {
                os << "  Node_" << i << " -> Node_" << it->second 
                   << " [color=\"#d62728\", style=\"dashed\", label=\"exec\", fontcolor=\"#d62728\"];\n";
            }
        }
    }

    os << "}\n";
    os.close();
    llvm::errs() << "Successfully dumped MemoryDependenceGraph to " << filename << "\n";
}

void ReorderOpsByBlockIdPass::runOnOperation() {
  LOG_DEBUG("\n=== Pass: TuningOpSeq ===\n");
  OpBuilder const builder(&getContext());

  auto moduleOp = getOperation();

  if (CVPipeline::hasFallbackAttr(moduleOp)) {
    return;
  }

  LOG_DEBUG("Input mlir:\n" << moduleOp << "\n");
  llvm::dbgs().flush();

  auto &aa = getAnalysis<AliasAnalysis>();
  auto memGraph = MemoryDependenceGraph(moduleOp, aa);
  auto bm = ComputeBlockIdManager(moduleOp);
  auto result = moduleOp.walk([&](Block *block) {
    auto *parentOp = block->getParentOp();
    if (!parentOp ||
        // whitelist ops to reorder
        !(isa<func::FuncOp>(parentOp) ||
          isa<scf::SCFDialect>(parentOp->getDialect()))) {
      return WalkResult::skip();
    }
    if (llvm::failed(reorderOpsInBlock(*block, memGraph, bm))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
    return;
  }

  LOG_DEBUG("Output mlir:\n" << moduleOp << "\n");
  LOG_DEBUG("=== Pass TuningOpSeq complete ===\n");
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createReorderOpsByBlockIdPass() {
  return std::make_unique<ReorderOpsByBlockIdPass>();
}
