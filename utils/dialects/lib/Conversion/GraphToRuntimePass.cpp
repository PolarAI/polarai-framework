//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Polar. All rights reserved.
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

#include "PolarGraph/PolarGraphDialect.h"
#include "PolarGraph/PolarGraphOps.h"
#include "PolarRuntime/PolarRuntimeDialect.h"
#include "PolarRuntime/PolarRuntimeOps.h"

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
class PolarGraphConversionPattern : public ConversionPattern {
public:
  PolarGraphConversionPattern(MLIRContext* context,
                              PatternBenefit patternBenefit = 1)
      : ConversionPattern(OpT::getOperationName(), patternBenefit, context) {}
};

template <typename OpT>
struct BuiltinConversionPattern : public PolarGraphConversionPattern<OpT> {
  using PolarGraphConversionPattern<OpT>::PolarGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto concreteOp = llvm::cast<OpT>(op);
    FuncOp node = concreteOp.template getParentOfType<FuncOp>();

    auto nodeIdAttr = node.getAttrOfType<mlir::IntegerAttr>(
        polar_graph::NodeOp::getNodeIdAttrName());
    auto deviceType = polar_rt::DeviceType::get(op->getContext());

    auto device = rewriter.create<polar_rt::DeviceSelectOp>(
        op->getLoc(), deviceType, nodeIdAttr);

    SmallVector<mlir::Type, 1> resTypes;
    resTypes.push_back(polar_rt::EventType::get(op->getContext()));

    // todo correctly deploy events
    // auto definingOp = operands.back().getDefiningOp();
    mlir::Value blockingEvent;
    // if (llvm::isa<polar_rt::LaunchOp>(definingOp)) {
    // blockingEvent = llvm::cast<polar_rt::LaunchOp>(definingOp).getResult(1);
    // } else {
    blockingEvent =
        rewriter.create<polar_rt::NullEventOp>(op->getLoc(), resTypes.back());
    // }

    // FIXME this pattern is incorrect if node performs more than one
    //       computation.
    auto applyOp = rewriter.create<polar_rt::ApplyOp>(
        op->getLoc(), device, blockingEvent, concreteOp.getKernelName(),
        operands);
    OpBuilder builder(op->getContext());
    builder.setInsertionPointToStart(&applyOp.body().front());
    concreteOp.produceKernel(builder, applyOp.body().front().getArguments());
    rewriter.eraseOp(op);
    return success();
  }
};

struct GraphReturnConversionPattern
    : public PolarGraphConversionPattern<polar_graph::ReturnOp> {
  using PolarGraphConversionPattern<
      polar_graph::ReturnOp>::PolarGraphConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    mlir::Value retVal;
    auto users = operands.front().getUsers();
    std::vector<Operation*> launches;
    std::copy_if(
        users.begin(), users.end(), std::back_inserter(launches),
        [](Operation* op) { return llvm::isa<polar_rt::ApplyOp>(op); });

    // fixme may be incorrect
    if (!launches.empty()) {
      auto launchOp = llvm::cast<polar_rt::ApplyOp>(launches.back());
      retVal = launchOp.getResult();
    } else {
      retVal = rewriter.create<polar_rt::NullEventOp>(
          op->getLoc(), polar_rt::EventType::get(op->getContext()));
    }
    rewriter.replaceOpWithNewOp<ReturnOp>(op, ValueRange{retVal});

    return success();
  }
};

struct AllocOpConversionPattern
    : public PolarGraphConversionPattern<polar_graph::AllocOp> {
  using PolarGraphConversionPattern<
      polar_graph::AllocOp>::PolarGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {

    auto concreteOp = llvm::cast<polar_graph::AllocOp>(op);
    auto node = concreteOp.template getParentOfType<FuncOp>();

    auto nodeIdAttr = node.getAttrOfType<mlir::IntegerAttr>(
        polar_graph::NodeOp::getNodeIdAttrName());
    auto deviceType = polar_rt::DeviceType::get(op->getContext());

    auto device = rewriter.create<polar_rt::DeviceSelectOp>(
        op->getLoc(), deviceType, nodeIdAttr);
    rewriter.replaceOpWithNewOp<polar_rt::AllocOp>(op, device, operands[0]);

    return success();
  }
};

struct ReleaseOpConversionPattern
    : public PolarGraphConversionPattern<polar_graph::ReleaseOp> {
  using PolarGraphConversionPattern<
      polar_graph::ReleaseOp>::PolarGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {

    auto concreteOp = llvm::cast<polar_graph::ReleaseOp>(op);
    auto node = concreteOp.getParentOfType<FuncOp>();

    auto nodeIdAttr = node.getAttrOfType<mlir::IntegerAttr>(
        polar_graph::NodeOp::getNodeIdAttrName());
    auto deviceType = polar_rt::DeviceType::get(op->getContext());

    auto device = rewriter.create<polar_rt::DeviceSelectOp>(
        op->getLoc(), deviceType, nodeIdAttr);
    rewriter.replaceOpWithNewOp<polar_rt::ReleaseOp>(op, device, operands[0],
                                                     ValueRange{});

    return success();
  }
};

struct LockOpConversionPattern
    : public PolarGraphConversionPattern<polar_graph::LockOp> {
  using PolarGraphConversionPattern<
      polar_graph::LockOp>::PolarGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {

    auto concreteOp = llvm::cast<polar_graph::LockOp>(op);
    auto node = concreteOp.getParentOfType<FuncOp>();

    auto nodeIdAttr = node.getAttrOfType<mlir::IntegerAttr>(
        polar_graph::NodeOp::getNodeIdAttrName());
    auto deviceType = polar_rt::DeviceType::get(op->getContext());

    auto device = rewriter.create<polar_rt::DeviceSelectOp>(
        op->getLoc(), deviceType, nodeIdAttr);
    rewriter.replaceOpWithNewOp<polar_rt::LockOp>(op, device, operands[0],
                                                  concreteOp.lock_type());

    return success();
  }
};

struct GraphTerminatorConversionPattern
    : public PolarGraphConversionPattern<polar_graph::GraphTerminatorOp> {
  using PolarGraphConversionPattern<
      polar_graph::GraphTerminatorOp>::PolarGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, operands);

    return success();
  }
};

struct NodeOpConversionPattern
    : public PolarGraphConversionPattern<polar_graph::NodeOp> {
  using PolarGraphConversionPattern<
      polar_graph::NodeOp>::PolarGraphConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto node = llvm::cast<polar_graph::NodeOp>(op);

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
        {polar_rt::GraphHandleType::get(op->getContext())},
        {polar_rt::EventType::get(op->getContext())});
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
    : public PolarGraphConversionPattern<polar_graph::GraphOp> {
  using PolarGraphConversionPattern<
      polar_graph::GraphOp>::PolarGraphConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto graph = llvm::cast<polar_graph::GraphOp>(op);

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
        {polar_rt::GraphHandleType::get(op->getContext())}, {});
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
    : public PolarGraphConversionPattern<polar_graph::EvalOp> {
  using PolarGraphConversionPattern<
      polar_graph::EvalOp>::PolarGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto evalOp = llvm::cast<polar_graph::EvalOp>(op);
    auto module = evalOp.getParentOfType<ModuleOp>();
    auto nodeFunc = module.lookupSymbol<FuncOp>(evalOp.node());
    auto parentFunc = evalOp.getParentOfType<FuncOp>();

    auto graphHandle = parentFunc.getArgument(0);

    rewriter.replaceOpWithNewOp<CallOp>(op, nodeFunc, ValueRange{graphHandle});
    return success();
  }
};

struct BarrierConversionPattern
    : PolarGraphConversionPattern<polar_graph::BarrierOp> {
  using PolarGraphConversionPattern<
      polar_graph::BarrierOp>::PolarGraphConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto barrierOp = llvm::cast<polar_graph::BarrierOp>(op);

    auto attr = barrierOp.clusterIdAttr();

    auto newBarrier =
        rewriter.create<polar_rt::BarrierOp>(op->getLoc(), ValueRange{});
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
    target.addLegalDialect<polar_rt::PolarRuntimeDialect>();
    target.addLegalDialect<polar_graph::PolarGraphDialect>();

    target.addIllegalOp<polar_graph::EvalOp>();
    target.addIllegalOp<polar_graph::NodeOp>();
    target.addIllegalOp<polar_graph::ReturnOp>();
    target.addIllegalOp<polar_graph::GraphTerminatorOp>();
    target.addIllegalOp<polar_graph::GraphOp>();
    target.addIllegalOp<polar_graph::BarrierOp>();
    target.addIllegalOp<polar_graph::AllocOp>();
    target.addIllegalOp<polar_graph::ReleaseOp>();
    target.addIllegalOp<polar_graph::LockOp>();
    target.addIllegalOp<polar_graph::AddOp>();
    target.addIllegalOp<polar_graph::Conv2DOp>();
    target.addIllegalOp<polar_graph::CopyOp>();
    target.addIllegalOp<polar_graph::DivideOp>();
    target.addIllegalOp<polar_graph::LogLossOp>();
    target.addIllegalOp<polar_graph::MulOp>();
    target.addIllegalOp<polar_graph::MulConcatOp>();
    target.addIllegalOp<polar_graph::MatMulOp>();
    target.addIllegalOp<polar_graph::Pool2DOp>();
    target.addIllegalOp<polar_graph::ReLUOp>();
    target.addIllegalOp<polar_graph::SigmoidOp>();
    target.addIllegalOp<polar_graph::SoftmaxOp>();
    target.addIllegalOp<polar_graph::TransposeOp>();
    target.addIllegalOp<polar_graph::TransposeOp>();
    target.addIllegalOp<polar_graph::FillOp>();

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
      BuiltinConversionPattern<polar_graph::AddOp>,
      BuiltinConversionPattern<polar_graph::Conv2DOp>,
      BuiltinConversionPattern<polar_graph::CopyOp>,
      BuiltinConversionPattern<polar_graph::DivideOp>,
      BuiltinConversionPattern<polar_graph::LogLossOp>,
      BuiltinConversionPattern<polar_graph::MulOp>,
      BuiltinConversionPattern<polar_graph::MulConcatOp>,
      BuiltinConversionPattern<polar_graph::MatMulOp>,
      BuiltinConversionPattern<polar_graph::Pool2DOp>,
      BuiltinConversionPattern<polar_graph::ReLUOp>,
      BuiltinConversionPattern<polar_graph::SigmoidOp>,
      BuiltinConversionPattern<polar_graph::SoftmaxOp>,
      BuiltinConversionPattern<polar_graph::FillOp>,
      BuiltinConversionPattern<polar_graph::TransposeOp>
      // clang-format on
      >(ctx);
}

auto createLowerGraphToRuntimePass()
    -> std::unique_ptr<OperationPass<ModuleOp>> {
  return std::make_unique<GraphToRuntimePass>();
}
} // namespace mlir
