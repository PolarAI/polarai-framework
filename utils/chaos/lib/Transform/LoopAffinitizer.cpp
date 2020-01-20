//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include "LoopAffinitizer.h"
#include <iostream>
#include <mlir/Analysis/Dominance.h>
#include <mlir/Dialect/AffineOps/AffineOps.h>
#include <mlir/Dialect/StandardOps/Ops.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IntegerSet.h>
#include <set>
#include <stack>
#include <vector>

using namespace mlir;

namespace chaos {

std::unique_ptr<LoopAffinitizer> createLoopAffinitizerPass() {
  return std::make_unique<LoopAffinitizer>();
}

using Loop = std::vector<Block*>;

std::vector<Block*> getLoopBody(Block* header, Block* tail) {
  if (header == tail)
    return {};
  std::vector<Block*> body;
  std::stack<Block*> stack;
  std::set<Block*> viewedBlocks;
  viewedBlocks.insert(header);
  viewedBlocks.insert(tail);

  for (auto* successor : header->getSuccessors()) {
    if (!viewedBlocks.count(successor))
      stack.push(successor);
  }

  while (!stack.empty()) {
    auto* block = stack.top();
    //    if (block != tail) {
    viewedBlocks.insert(block);
    body.push_back(block);
    for (auto* successor : block->getSuccessors()) {
      if (!viewedBlocks.count(block))
        stack.push(successor);
    }

    stack.pop();
    //    }
  }

  return body;
}

static bool isLoopHeader(std::vector<Loop>& loops, void* block) {
  for (auto& loop : loops) {
    if (loop.front() == block) {
      return true;
    }
  }
  return false;
}

static bool isLoopFooter(std::vector<Loop>& loops, void* block) {
  for (auto& loop : loops) {
    if (loop.back() == block) {
      return true;
    }
  }
  return false;
}

static std::tuple<AffineForOp, Block*>
createLoop(OpBuilder builder, Loop& loop, MLIRContext* ctx,
           BlockAndValueMapping mapping) {
  //  Loop innermostLoop = getInnermostLoop(dominanceInfo, loops);
  std::clog << "Loop blocks count: " << loop.size() << std::endl;

  auto* header = loop.front();
  auto* footer = loop.back();

  auto condBr = dyn_cast_or_null<CondBranchOp>(&footer->back());
  if (!condBr) {
    //    signalPassFailure(); // todo err message
    return {nullptr, nullptr};
  }
  if (condBr.getNumOperands() != 2) {
    //    signalPassFailure();
    return {nullptr, nullptr};
  }

  auto cmpOp = dyn_cast_or_null<CmpIOp>(condBr.getOperand(0)->getDefiningOp());
  auto addOp = dyn_cast_or_null<AddIOp>(condBr.getOperand(1)->getDefiningOp());

  if (!cmpOp || !addOp) {
    //    signalPassFailure();
    return {nullptr, nullptr};
  }

  std::cout << "cmp args " << cmpOp.getNumOperands() << std::endl;

  auto upperBound = mapping.lookup(cmpOp.getOperand(1));

  auto lowerBoundConst =
      builder.create<ConstantIntOp>(header->front().getLoc(), 0, 64);
  auto lowerBoundValue = lowerBoundConst.getResult();

  auto map = AffineMap::get(1, 1, {getAffineSymbolExpr(0, ctx)});

  auto affineLoop = builder.create<AffineForOp>(
      header->front().getLoc(), lowerBoundValue, map, *upperBound, map, 1);

  return {affineLoop, footer->getSuccessor(0)};
}

static Loop& getLoop(Block* header, std::vector<Loop>& loops) {
  for (auto& loop : loops) {
    if (loop.front() == header) {
      return loop;
    }
  }
  llvm_unreachable("Loop was not found");
}

static Block* reduceBlock(Block* block) {
  if (block->getNumSuccessors() == 1) {
    if (block->front().getName().getStringRef() == "std.br") {
      return block->getSuccessor(0);
    }
  }
  return block;
}

static void copyBlock(OpBuilder builder, Block* srcBlock,
                      BlockAndValueMapping mapping, std::vector<Loop>& loops) {
  if (isLoopHeader(loops, srcBlock)) {
    Loop& loop = getLoop(srcBlock, loops);

    auto [loopOp, nextBlock] =
        createLoop(builder, loop, builder.getContext(), mapping);
    builder.setInsertionPointToStart(loopOp.getBody());
    auto castOp = builder.create<IndexCastOp>(
        builder.getUnknownLoc(), loopOp.getInductionVar(),
        srcBlock->getArgument(0).getType());
    mapping.map(srcBlock->getArgument(0), castOp);

    if (loop.size() == 1) {
      auto* body = loop.front();
      auto blockEnd = body->end();
      builder.setInsertionPointToStart(loopOp.getBody());

      --blockEnd; // omit block terminator
      for (auto it = body->begin(); it != blockEnd; it++) {
        builder.clone(*it, mapping);
      }
    } else {
      auto* body = loop[1];
      builder.setInsertionPointToStart(loopOp.getBody());
      copyBlock(builder, body, mapping, loops);
    }

    builder.setInsertionPointAfter(loopOp);
    auto reduced = reduceBlock(nextBlock);
    bool shouldCopy = !isLoopFooter(loops, reduced);
    shouldCopy &= !(loopOp.getParentOp()->getName().getStringRef() != "func" &&
                    reduced->getNumSuccessors() == 0);
    if (shouldCopy) {
      for (auto& blockArg : reduced->getArguments()) {
        if (auto definingOp = blockArg.getDefiningOp()) {
          auto to = mapping.lookup(definingOp->getResult(0));
          mapping.map(blockArg, to);
        }
      }
      copyBlock(builder, reduced, mapping, loops);
    }

    return;
  } else if (isLoopFooter(loops, srcBlock)) {
    copyBlock(builder, srcBlock->getTerminator()->getSuccessor(0), mapping,
              loops);
    return;
  }

  for (auto& op : *srcBlock) {
    if (op.getName().getStringRef() == "std.cond_br") {
      auto predicate = mapping.lookup(op.getOperand(0));
      auto indexPredicate =
          builder.create<IndexCastOp>(builder.getUnknownLoc(), predicate,
                                      IndexType::get(builder.getContext()));

      ValueRange range{indexPredicate};

      auto sub = getAffineBinaryOpExpr(
          AffineExprKind::Add, getAffineSymbolExpr(0, op.getContext()),
          getAffineConstantExpr(-1, op.getContext()));
      auto set = IntegerSet::get(0, 1, {sub}, {true});
      auto ifOp = builder.create<AffineIfOp>(op.getLoc(), set, range, false);
      copyBlock(ifOp.getThenBodyBuilder(), op.getSuccessor(1), mapping, loops);
      copyBlock(builder, op.getSuccessor(0), mapping, loops);
      ifOp.verify();
    } else if (op.getName().getStringRef() == "std.br") {
      auto dstBlock = op.getSuccessor(0);
      for (auto& blockArg : dstBlock->getArguments()) {
        if (auto definingOp = blockArg.getDefiningOp()) {
          auto to = mapping.lookup(definingOp->getResult(0));
          mapping.map(blockArg, to);
        }
      }
      copyBlock(builder, op.getSuccessor(0), mapping, loops);
    } else {
      auto resOp = builder.clone(op, mapping);
      if (op.getNumResults() == 1) {
        mapping.map(op.getResult(0), resOp->getResult(0));
      }
    }
  }
}

static void rewriteFunction(FuncOp& oldFunc, std::vector<Loop>& loops) {
  OpBuilder builder(oldFunc);
  auto newFunc = builder.create<FuncOp>(oldFunc.getLoc(), oldFunc.getName(),
                                        oldFunc.getType(), oldFunc.getAttrs());
  auto* funcBlock = newFunc.addEntryBlock();
  builder.setInsertionPointToStart(funcBlock);
  BlockAndValueMapping mapping;
  for (size_t argId = 0; argId < oldFunc.getNumArguments(); argId++) {
    mapping.map(oldFunc.getArgument(argId), newFunc.getArgument(argId));
  }
  copyBlock(builder, &oldFunc.getBody().front(), mapping, loops);
  oldFunc.setName(oldFunc.getName().str() + "_wasted");
  oldFunc.setAttr("safe_to_remove", BoolAttr::get(true, oldFunc.getContext()));
  newFunc.verify();
}

void LoopAffinitizer::runOnOperation() {
  FuncOp func = getOperation();

  if (func.getBody().empty()) {
    return;
  }

  DominanceInfo dominanceInfo(func);

  std::vector<Loop> loops;

  for (auto& block : func) {
    //    std::cout << &block << std::endl;
    //    auto node = dominanceInfo.getNode(&block);
    //    std::cout << node->getNumChildren() << std::endl;
    for (auto& op : block) {
      std::cerr << "op name: " << op.getName().getStringRef().data()
                << std::endl;
      if (op.getName().getStringRef() == "std.cond_br") {
        if ( //&block != op.getSuccessor(1) &&
            dominanceInfo.dominates(op.getSuccessor(1), &block)) {
          // Now we found a back edge.
          std::cout << "successors " << op.getSuccessor(1)->getNumSuccessors()
                    << std::endl;
          Loop loop;

          auto header = op.getSuccessor(1);
          loop.push_back(header);

          auto body = getLoopBody(header, &block);
          std::copy(body.begin(), body.end(), std::back_inserter(loop));

          if (header != &block)
            loop.push_back(&block);

          loops.push_back(loop);
        }
      }
    }
  }

  rewriteFunction(func, loops);
}
} // namespace chaos
