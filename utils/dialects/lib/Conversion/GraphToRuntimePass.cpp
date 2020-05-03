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

#include "Conversion/GraphToRuntimePass.h"
#include "AthenaGraph/AthenaGraphOps.h"
#include "AthenaRuntime/AthenaRuntimeDialect.h"
#include "AthenaRuntime/AthenaRuntimeOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

template <typename OpT>
struct BuiltinToFuncCallLoweringPattern : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter& rewriter) const override {
    auto* llvmDialect =
        op.getContext()->template getRegisteredDialect<LLVM::LLVMDialect>();
    auto parentFunc = op.template getParentOfType<FuncOp>();
    auto device = parentFunc.getArgument(parentFunc.getNumArguments() - 1);
    auto allocator = parentFunc.getArgument(parentFunc.getNumArguments() - 2);

    op.dump();
    auto devVoidPtr = rewriter.create<ath_rt::AnyCastOp>(
        op.getLoc(), LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo(),
        device);
    auto allocVoidPtr = rewriter.create<ath_rt::AnyCastOp>(
        op.getLoc(), LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo(),
        allocator);

    SmallVector<Value, 3> operands;
    operands.push_back(devVoidPtr);
    operands.push_back(allocVoidPtr);
    if constexpr (std::is_same_v<OpT, ath_graph::GetTensor>) {
      auto tensorAddrI64 = rewriter.create<ath_rt::AnyCastOp>(
          op.getLoc(), LLVM::LLVMType::getInt64Ty(llvmDialect),
          op.getOperand());
      operands.push_back(tensorAddrI64);
    } else {
      auto tensorVoidPtr = rewriter.create<ath_rt::AnyCastOp>(
          op.getLoc(), LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo(),
          op.getOperand());
      operands.push_back(tensorVoidPtr);
    }

    if constexpr (std::is_same_v<OpT, ath_graph::LockOp>) {
      auto lockType =
          op.template getAttrOfType<StringAttr>("lock_type").getValue();
      int lockTypeInt;
      if (lockType == "read") {
        lockTypeInt = 0;
      } else {
        lockTypeInt = 1;
      }

      auto typeConst = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
          rewriter.getIntegerAttr(LLVM::LLVMType::getInt32Ty(llvmDialect),
                                  lockTypeInt));
      operands.push_back(typeConst);
    }

    std::string symbolName;
    if constexpr (std::is_same_v<OpT, ath_graph::AllocOp>) {
      symbolName = "ath_allocate";
    } else if constexpr (std::is_same_v<OpT, ath_graph::ReleaseOp>) {
      symbolName = "ath_release_tensor";
    } else if constexpr (std::is_same_v<OpT, ath_graph::GetTensor>) {
      symbolName = "ath_get_tensor_ptr";
    } else if constexpr (std::is_same_v<OpT, ath_graph::LockOp>) {
      symbolName = "ath_lock_tensor";
    }

    auto funcOp = op.template getParentOfType<ModuleOp>()
                      .template lookupSymbol<LLVM::LLVMFuncOp>(symbolName);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp, operands);
    return success();
  }
};

struct InvokeLoaderLowering
    : public OpRewritePattern<ath_graph::InvokeLoaderOp> {
  using OpRewritePattern<ath_graph::InvokeLoaderOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ath_graph::InvokeLoaderOp op,
                                PatternRewriter& rewriter) const override {
    auto* llvmDialect =
        op.getContext()->template getRegisteredDialect<LLVM::LLVMDialect>();
    auto parentFunc = op.template getParentOfType<FuncOp>();
    auto device = parentFunc.getArgument(parentFunc.getNumArguments() - 1);
    auto allocator = parentFunc.getArgument(parentFunc.getNumArguments() - 2);

    auto devVoidPtr = rewriter.create<ath_rt::AnyCastOp>(
        op.getLoc(), LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo(),
        device);
    auto allocVoidPtr = rewriter.create<ath_rt::AnyCastOp>(
        op.getLoc(), LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo(),
        allocator);

    SmallVector<Value, 3> operands;
    operands.push_back(allocVoidPtr);
    operands.push_back(devVoidPtr); // FIXME this must be a loader. Currently no
                                    //   way to express this with Athena Graph.
    auto tensorVoidPtr = rewriter.create<ath_rt::AnyCastOp>(
        op.getLoc(), LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo(),
        op.getOperand());
    operands.push_back(tensorVoidPtr);

    auto funcOp =
        op.template getParentOfType<ModuleOp>()
            .template lookupSymbol<LLVM::LLVMFuncOp>(op.loader_routine());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp, operands);

    return success();
  }
};

template <typename OpT>
class BuiltinConversionPattern : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter& rewriter) const override {
    rewriter.create<mlir::ath_rt::LaunchOp>(op.getLoc(), op.getOperationName(),
                                            op.getOperands());
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename FuncT>
class FunctionConversionPattern : public OpRewritePattern<FuncT> {
public:
  using OpRewritePattern<FuncT>::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncT op,
                                PatternRewriter& rewriter) const override {
    auto oldType = op.getType();
    SmallVector<Type, 5> newArgs;
    newArgs.append(oldType.getInputs().begin(), oldType.getInputs().end());
    // Allocator pointer
    newArgs.push_back(rewriter.getIndexType());
    if constexpr (std::is_same_v<FuncT, ath_graph::NodeOp>) {
      // Device pointer
      newArgs.push_back(rewriter.getIndexType());
    }
    auto newFuncType = rewriter.getFunctionType(newArgs, oldType.getResults());
    auto newFunc = rewriter.create<mlir::FuncOp>(op.getLoc(), op.getName(),
                                                 newFuncType, op.getAttrs());

    auto& oldBodyFront = op.getBody().front();
    oldBodyFront.insertArgument(oldBodyFront.getArguments().end(),
                                rewriter.getIndexType());
    if constexpr (std::is_same_v<FuncT, ath_graph::NodeOp>) {
      // Device pointer
      oldBodyFront.insertArgument(oldBodyFront.getArguments().end(),
                                  rewriter.getIndexType());
    }

    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(),
                                newFunc.getBody().end());
    rewriter.eraseOp(op);

    return success();
  }
};

class SliceLoweringPattern : public OpRewritePattern<ath_graph::SliceOp> {
public:
  using OpRewritePattern<ath_graph::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ath_graph::SliceOp op,
                                PatternRewriter& rewriter) const override {
    auto* llvmDialect =
        op.getContext()->template getRegisteredDialect<LLVM::LLVMDialect>();

    auto index = rewriter.create<ath_rt::AnyCastOp>(
        op.getLoc(), LLVM::LLVMType::getInt64Ty(llvmDialect), op.getOperand(0));
    auto tensor = rewriter.create<ath_rt::AnyCastOp>(
        op.getLoc(), LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo(),
        op.getOperand(1));

    auto funcOp =
        op.template getParentOfType<ModuleOp>()
            .template lookupSymbol<LLVM::LLVMFuncOp>("ath_get_sub_tensor");

    SmallVector<Value, 2> operands;
    operands.push_back(index);
    operands.push_back(tensor);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp, operands);
    return success();
  }
};

class EvalLoweringPattern : public OpRewritePattern<ath_graph::EvalOp> {
public:
  using OpRewritePattern<ath_graph::EvalOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ath_graph::EvalOp op,
                                PatternRewriter& rewriter) const override {
    auto* llvmDialect =
        op.getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    auto parentFunc = op.getParentOfType<FuncOp>();
    auto context = parentFunc.getArgument(parentFunc.getNumArguments() - 3);
    auto allocator = parentFunc.getArgument(parentFunc.getNumArguments() - 2);
    auto getDevFunc =
        op.getParentOfType<ModuleOp>().lookupSymbol<LLVM::LLVMFuncOp>(
            "ath_get_device_for_node");

    auto node = op.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(op.node());
    auto nodeIdAttr =
        node.getAttrOfType<IntegerAttr>(ath_graph::NodeOp::getNodeIdAttrName());

    auto nodeId = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), LLVM::LLVMType::getInt64Ty(llvmDialect),
        rewriter.getIntegerAttr(LLVM::LLVMType::getInt64Ty(llvmDialect),
                                nodeIdAttr.getValue()));
    auto ctxPtr = rewriter.create<ath_rt::AnyCastOp>(
        op.getLoc(), LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo(),
        context);

    SmallVector<Value, 2> operands;
    operands.push_back(nodeId);
    operands.push_back(ctxPtr);

    auto devPtr =
        rewriter.create<LLVM::CallOp>(op.getLoc(), getDevFunc, operands);

    SmallVector<Value, 5> nodeArgs;
    std::copy(op.getArgOperands().begin(), op.getArgOperands().end(),
              std::back_inserter(nodeArgs));
    nodeArgs.push_back(allocator);
    auto devIdx = rewriter.create<ath_rt::AnyCastOp>(
        op.getLoc(), rewriter.getIndexType(), devPtr.getResult(0));
    nodeArgs.push_back(devIdx);

    rewriter.replaceOpWithNewOp<CallOp>(op, node, nodeArgs);

    return success();
  }
};

class ReturnLoweringPattern : public OpRewritePattern<ath_graph::ReturnOp> {
public:
  using OpRewritePattern<ath_graph::ReturnOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ath_graph::ReturnOp op,
                                PatternRewriter& rewriter) const override {
    auto cast = rewriter.create<ath_rt::AnyCastOp>(
        op.getLoc(), op.getParentOfType<FuncOp>().getCallableResults().front(),
        op.getOperand(0));
    rewriter.replaceOpWithNewOp<ReturnOp>(op, ValueRange{cast});
    return success();
  }
};

class GraphTerminatorLoweringPattern
    : public OpRewritePattern<ath_graph::GraphTerminatorOp> {
public:
  using OpRewritePattern<ath_graph::GraphTerminatorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ath_graph::GraphTerminatorOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return success();
  }
};

namespace {
class LowerGraphPass
    : public PassWrapper<LowerGraphPass, OperationPass<ModuleOp>> {
protected:
  void runOnOperation() override {
    OwningRewritePatternList structureLoweringPatterns, nodeOpsLoweringPatterns;
    populateGraphToRuntimeConversionPatterns(
        structureLoweringPatterns, nodeOpsLoweringPatterns, &getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect, ath_rt::AthenaRuntimeDialect,
                           LLVM::LLVMDialect>();
    target.addLegalOp<FuncOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      structureLoweringPatterns))) {
      signalPassFailure();
    }
    if (failed(applyPartialConversion(getOperation(), target,
                                      nodeOpsLoweringPatterns))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
void populateGraphToRuntimeConversionPatterns(
    OwningRewritePatternList& structureLoweringPatterns,
    OwningRewritePatternList& nodeOpsLoweringPatterns, MLIRContext* ctx) {
  structureLoweringPatterns
      .insert<ReturnLoweringPattern, GraphTerminatorLoweringPattern,
              FunctionConversionPattern<ath_graph::NodeOp>,
              FunctionConversionPattern<ath_graph::GraphOp>>(ctx);
  nodeOpsLoweringPatterns.insert<
      // clang-format off
      BuiltinToFuncCallLoweringPattern<ath_graph::AllocOp>,
      BuiltinToFuncCallLoweringPattern<ath_graph::ReleaseOp>,
      BuiltinToFuncCallLoweringPattern<ath_graph::GetTensor>,
      BuiltinToFuncCallLoweringPattern<ath_graph::LockOp>,
      InvokeLoaderLowering,
      SliceLoweringPattern,
      EvalLoweringPattern,
      BuiltinConversionPattern<ath_graph::AddOp>,
      BuiltinConversionPattern<ath_graph::MulOp>,
      BuiltinConversionPattern<ath_graph::MatmulOp>,
      BuiltinConversionPattern<ath_graph::TransposeOp>,
      BuiltinConversionPattern<ath_graph::FillOp>
      // clang-format on
      >(ctx);
}
std::unique_ptr<OperationPass<ModuleOp>> createLowerGraphToRuntimePass() {
  return std::make_unique<LowerGraphPass>();
}
} // namespace mlir
