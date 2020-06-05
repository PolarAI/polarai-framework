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

#include "AthenaGraph/AthenaGraphDialect.h"
#include "AthenaGraph/AthenaGraphOps.h"
#include "AthenaRuntime/AthenaRuntimeDialect.h"
#include "AthenaRuntime/AthenaRuntimeOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/IRBuilder.h"

using namespace mlir;

namespace {
template <typename OpT>
class AthenaGraphConversionPattern : public ConversionPattern {
public:
  AthenaGraphConversionPattern(MLIRContext* context,
                               PatternBenefit patternBenefit = 1)
      : ConversionPattern(OpT::getOperationName(), patternBenefit, context) {}
};

template <typename OpT>
struct BuiltinConversionPattern : public AthenaGraphConversionPattern<OpT> {
  using AthenaGraphConversionPattern<OpT>::AthenaGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto concreteOp = llvm::cast<OpT>(op);
    FuncOp node = concreteOp.template getParentOfType<FuncOp>();

    auto nodeIdAttr = node.getAttrOfType<mlir::IntegerAttr>(
        ath_graph::NodeOp::getNodeIdAttrName());
    auto deviceType = ath_rt::DeviceType::get(op->getContext());

    auto device = rewriter.create<ath_rt::DeviceSelectOp>(
        op->getLoc(), deviceType, nodeIdAttr);

    SmallVector<mlir::Type, 2> resTypes;
    resTypes.push_back(operands.back().getType());
    resTypes.push_back(ath_rt::EventType::get(op->getContext()));

    auto definingOp = operands.back().getDefiningOp();
    mlir::Value blockingEvent;
    if (llvm::isa<ath_rt::LaunchOp>(definingOp)) {
      blockingEvent = llvm::cast<ath_rt::LaunchOp>(definingOp).getResult(1);
    } else {
      blockingEvent =
          rewriter.create<ath_rt::NullEventOp>(op->getLoc(), resTypes.back());
    }

    RankedTensorType tensorType =
        concreteOp.getResult().getType().template cast<RankedTensorType>();
    auto globalSize = rewriter.getI64ArrayAttr(tensorType.getShape());
    SmallVector<int64_t, 3> local(tensorType.getRank());
    auto localSize = rewriter.getI64ArrayAttr(local);

    // FIXME this pattern is incorrect if node performs more than one
    //       computation.
    auto launchOp = rewriter.create<ath_rt::LaunchOp>(
        op->getLoc(), resTypes, device, blockingEvent,
        concreteOp.getKernelName(), globalSize, localSize, operands);
    rewriter.replaceOp(op, launchOp.getResult(0));
    return success();
  }
};

struct GraphReturnConversionPattern
    : public AthenaGraphConversionPattern<ath_graph::ReturnOp> {
  using AthenaGraphConversionPattern<
      ath_graph::ReturnOp>::AthenaGraphConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    mlir::Value retVal;
    auto definingOp = operands.front().getDefiningOp();
    if (llvm::isa<ath_rt::LaunchOp>(definingOp)) {
      auto launchOp = llvm::cast<ath_rt::LaunchOp>(definingOp);
      retVal = launchOp.getResult(1);
    } else {
      retVal = rewriter.create<ath_rt::NullEventOp>(
          op->getLoc(), ath_rt::EventType::get(op->getContext()));
    }
    rewriter.replaceOpWithNewOp<ReturnOp>(op, ValueRange{retVal});

    return success();
  }
};

struct AllocOpConversionPattern
    : public AthenaGraphConversionPattern<ath_graph::AllocOp> {
  using AthenaGraphConversionPattern<
      ath_graph::AllocOp>::AthenaGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {

    auto concreteOp = llvm::cast<ath_graph::AllocOp>(op);
    auto node = concreteOp.template getParentOfType<FuncOp>();

    auto nodeIdAttr = node.getAttrOfType<mlir::IntegerAttr>(
        ath_graph::NodeOp::getNodeIdAttrName());
    auto deviceType = ath_rt::DeviceType::get(op->getContext());

    auto device = rewriter.create<ath_rt::DeviceSelectOp>(
        op->getLoc(), deviceType, nodeIdAttr);
    rewriter.replaceOpWithNewOp<ath_rt::AllocOp>(op, device, operands[0]);

    return success();
  }
};

struct ReleaseOpConversionPattern
    : public AthenaGraphConversionPattern<ath_graph::ReleaseOp> {
  using AthenaGraphConversionPattern<
      ath_graph::ReleaseOp>::AthenaGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {

    auto concreteOp = llvm::cast<ath_graph::ReleaseOp>(op);
    auto node = concreteOp.getParentOfType<FuncOp>();

    auto nodeIdAttr = node.getAttrOfType<mlir::IntegerAttr>(
        ath_graph::NodeOp::getNodeIdAttrName());
    auto deviceType = ath_rt::DeviceType::get(op->getContext());

    auto device = rewriter.create<ath_rt::DeviceSelectOp>(
        op->getLoc(), deviceType, nodeIdAttr);
    rewriter.replaceOpWithNewOp<ath_rt::ReleaseOp>(op, device, operands[0],
                                                   ValueRange{});

    return success();
  }
};

struct LockOpConversionPattern
    : public AthenaGraphConversionPattern<ath_graph::LockOp> {
  using AthenaGraphConversionPattern<
      ath_graph::LockOp>::AthenaGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {

    auto concreteOp = llvm::cast<ath_graph::LockOp>(op);
    auto node = concreteOp.getParentOfType<FuncOp>();

    auto nodeIdAttr = node.getAttrOfType<mlir::IntegerAttr>(
        ath_graph::NodeOp::getNodeIdAttrName());
    auto deviceType = ath_rt::DeviceType::get(op->getContext());

    auto device = rewriter.create<ath_rt::DeviceSelectOp>(
        op->getLoc(), deviceType, nodeIdAttr);
    rewriter.replaceOpWithNewOp<ath_rt::LockOp>(op, device, operands[0],
                                                concreteOp.lock_type());

    return success();
  }
};

struct GraphTerminatorConversionPattern
    : public AthenaGraphConversionPattern<ath_graph::GraphTerminatorOp> {
  using AthenaGraphConversionPattern<
      ath_graph::GraphTerminatorOp>::AthenaGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, operands);

    return success();
  }
};

struct NodeOpConversionPattern
    : public AthenaGraphConversionPattern<ath_graph::NodeOp> {
  using AthenaGraphConversionPattern<
      ath_graph::NodeOp>::AthenaGraphConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto node = llvm::cast<ath_graph::NodeOp>(op);

    auto allAttrs = node.getAttrs();
    SmallVector<mlir::NamedAttribute, 4> newAttrs(allAttrs.begin(),
                                                  allAttrs.end());
    auto rem = std::remove_if(newAttrs.begin(), newAttrs.end(),
                              [&](mlir::NamedAttribute& attr) {
                                return attr.first == node.getTypeAttrName() ||
                                       attr.first == "sym_name";
                              });

    newAttrs.erase(rem, newAttrs.end());

    auto funcType = rewriter.getFunctionType(
        {ath_rt::GraphHandleType::get(op->getContext())},
        {ath_rt::EventType::get(op->getContext())});
    auto func = rewriter.create<FuncOp>(node.getLoc(), node.getName(), funcType,
                                        newAttrs);

    TypeConverter::SignatureConversion newSignature(0);
    newSignature.addInputs(funcType.getInput(0));

    rewriter.inlineRegionBefore(node.getBody(), func.getBody(),
                                func.getBody().end());
    rewriter.applySignatureConversion(&func.getBody(), newSignature);
    rewriter.eraseOp(op);

    return success();
  };
};

struct GraphOpConversionPattern
    : public AthenaGraphConversionPattern<ath_graph::GraphOp> {
  using AthenaGraphConversionPattern<
      ath_graph::GraphOp>::AthenaGraphConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto graph = llvm::cast<ath_graph::GraphOp>(op);

    auto allAttrs = graph.getAttrs();
    SmallVector<mlir::NamedAttribute, 4> newAttrs(allAttrs.begin(),
                                                  allAttrs.end());
    auto rem = std::remove_if(newAttrs.begin(), newAttrs.end(),
                              [&](mlir::NamedAttribute& attr) {
                                return attr.first == graph.getTypeAttrName() ||
                                       attr.first == "sym_name";
                              });

    newAttrs.erase(rem, newAttrs.end());
    auto funcType = rewriter.getFunctionType(
        {ath_rt::GraphHandleType::get(op->getContext())}, {});
    auto func = rewriter.create<FuncOp>(graph.getLoc(), graph.getName(),
                                        funcType, newAttrs);

    TypeConverter::SignatureConversion newSignature(0);
    newSignature.addInputs(funcType.getInput(0));

    rewriter.inlineRegionBefore(graph.body(), func.getBody(),
                                func.getBody().end());
    rewriter.applySignatureConversion(&func.getBody(), newSignature);
    rewriter.eraseOp(op);

    return success();
  };
};

struct EvalOpConversionPattern
    : public AthenaGraphConversionPattern<ath_graph::EvalOp> {
  using AthenaGraphConversionPattern<
      ath_graph::EvalOp>::AthenaGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto evalOp = llvm::cast<ath_graph::EvalOp>(op);
    auto module = evalOp.getParentOfType<ModuleOp>();
    auto nodeFunc = module.lookupSymbol<FuncOp>(evalOp.node());
    auto parentFunc = evalOp.getParentOfType<FuncOp>();

    auto graphHandle = parentFunc.getArgument(0);

    rewriter.replaceOpWithNewOp<CallOp>(op, nodeFunc, ValueRange{graphHandle});
    return success();
  }
};

struct BarrierConversionPattern
    : AthenaGraphConversionPattern<ath_graph::BarrierOp> {
  using AthenaGraphConversionPattern<
      ath_graph::BarrierOp>::AthenaGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto barrierOp = llvm::cast<ath_graph::BarrierOp>(op);

    auto attr = barrierOp.clusterIdAttr();

    auto newBarrier =
        rewriter.create<ath_rt::BarrierOp>(op->getLoc(), ValueRange{});
    newBarrier.setAttr("cluster_id", attr); // fixme refactor name
    rewriter.eraseOp(op);

    return success();
  }
};

class GraphToRuntimePass
    : public PassWrapper<GraphToRuntimePass, OperationPass<ModuleOp>> {
protected:
  void runOnOperation() {
    OwningRewritePatternList patterns;
    populateGraphToRuntimeConversionPatterns(patterns, &getContext());
    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<ModuleTerminatorOp>();
    target.addLegalOp<FuncOp>();
    target.addLegalOp<ReturnOp>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<ath_rt::AthenaRuntimeDialect>();
    target.addLegalDialect<ath_graph::AthenaGraphDialect>();

    target.addIllegalOp<ath_graph::EvalOp>();
    target.addIllegalOp<ath_graph::NodeOp>();
    target.addIllegalOp<ath_graph::ReturnOp>();
    target.addIllegalOp<ath_graph::GraphTerminatorOp>();
    target.addIllegalOp<ath_graph::GraphOp>();
    target.addIllegalOp<ath_graph::BarrierOp>();
    target.addIllegalOp<ath_graph::AllocOp>();
    target.addIllegalOp<ath_graph::ReleaseOp>();
    target.addIllegalOp<ath_graph::LockOp>();
    target.addIllegalOp<ath_graph::AddOp>();
    target.addIllegalOp<ath_graph::MulOp>();
    target.addIllegalOp<ath_graph::MatmulOp>();
    target.addIllegalOp<ath_graph::TransposeOp>();
    target.addIllegalOp<ath_graph::FillOp>();

    if (failed(applyPartialConversion(getOperation(), target, patterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
void populateGraphToRuntimeConversionPatterns(
    OwningRewritePatternList& loweringPatterns, MLIRContext* ctx) {
  loweringPatterns.insert<
      // clang-format off
      GraphOpConversionPattern,
      NodeOpConversionPattern,
      GraphTerminatorConversionPattern, 
      GraphReturnConversionPattern,
      EvalOpConversionPattern,
      BarrierConversionPattern,
      AllocOpConversionPattern,
      LockOpConversionPattern,
      ReleaseOpConversionPattern,
      BuiltinConversionPattern<ath_graph::AddOp>,
      BuiltinConversionPattern<ath_graph::MulOp>,
      BuiltinConversionPattern<ath_graph::MatmulOp>,
      BuiltinConversionPattern<ath_graph::FillOp>,
      BuiltinConversionPattern<ath_graph::TransposeOp>
      // clang-format on
      >(ctx);
}

auto createLowerGraphToRuntimePass()
    -> std::unique_ptr<OperationPass<ModuleOp>> {
  return std::make_unique<GraphToRuntimePass>();
}
} // namespace mlir
