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
#include "../utils/LaunchCommand.h"
#include "ArgInfo.h"
#include "AthenaGraph/AthenaGraphDialect.h"
#include "AthenaGraph/AthenaGraphOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

namespace mlir {
class AthenaTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  explicit AthenaTypeConverter(MLIRContext* ctx)
      : mLlvmDialect(ctx->getRegisteredDialect<LLVM::LLVMDialect>()) {
    mModule = &mLlvmDialect->getLLVMModule();
    mPointerWidth = mModule->getDataLayout().getPointerSizeInBits();

    addConversion([&](FloatType type) { return convertFloatType(type); });
    addConversion([&](IndexType type) { return convertIndexType(type); });
    addConversion([&](IntegerType type) { return convertIntegerType(type); });
    addConversion([&](TensorType type) { return convertTensorType(type); });

    addConversion([](LLVM::LLVMType type) { return type; });
    // Function types are handled separately.
    addConversion([](FunctionType type) { return type; });
  }

  MLIRContext& getContext() { return *getDialect()->getContext(); }
  llvm::LLVMContext& getLLVMContext() { return mModule->getContext(); }
  LLVM::LLVMDialect* getDialect() { return mLlvmDialect; }
  size_t getPointerWidth() { return mPointerWidth; }

  LLVM::LLVMType getVoidPtrTy() {
    return LLVM::LLVMType::getInt8Ty(mLlvmDialect).getPointerTo();
  }

protected:
  llvm::Module* mModule;
  LLVM::LLVMDialect* mLlvmDialect;
  size_t mPointerWidth;

private:
  auto convertIndexType(IndexType type) -> Type {
    return LLVM::LLVMType::getIntNTy(mLlvmDialect, mPointerWidth);
  }

  auto convertIntegerType(IntegerType type) -> Type {
    return LLVM::LLVMType::getIntNTy(mLlvmDialect, type.getWidth());
  }

  auto convertFloatType(FloatType type) -> Type {
    switch (type.getKind()) {
    case mlir::StandardTypes::F32:
      return LLVM::LLVMType::getFloatTy(mLlvmDialect);
    case mlir::StandardTypes::F64:
      return LLVM::LLVMType::getDoubleTy(mLlvmDialect);
    case mlir::StandardTypes::F16:
      return LLVM::LLVMType::getHalfTy(mLlvmDialect);
    default:
      llvm_unreachable("non-float type in convertFloatType");
    }
  }

  auto convertTensorType(TensorType type) -> Type { return getVoidPtrTy(); }
};
} // namespace mlir

template <typename OpT>
class AthenaConversionPattern : public ConversionPattern {
public:
  AthenaConversionPattern(AthenaTypeConverter& typeConverter,
                          PatternBenefit patternBenefit = 1)
      : ConversionPattern(OpT::getOperationName(), patternBenefit,
                          &typeConverter.getContext()),
        mTypeConverter(typeConverter) {}

protected:
  AthenaTypeConverter& mTypeConverter;
};

template <typename OpT>
struct BuiltinToFuncCallLoweringPattern : public AthenaConversionPattern<OpT> {
  using AthenaConversionPattern<OpT>::AthenaConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto castOp = llvm::cast<OpT>(op);
    auto* llvmDialect =
        op->getContext()->template getRegisteredDialect<LLVM::LLVMDialect>();
    auto parentFunc = op->template getParentOfType<LLVM::LLVMFuncOp>();
    auto device = parentFunc.getArgument(parentFunc.getNumArguments() - 1);
    auto allocator = parentFunc.getArgument(parentFunc.getNumArguments() - 2);

    SmallVector<Value, 3> builtinOperands;
    builtinOperands.push_back(device);
    builtinOperands.push_back(allocator);
    builtinOperands.push_back(castOp.getOperand());

    if constexpr (std::is_same_v<OpT, ath_graph::LockOp>) {
      auto lockType =
          op->template getAttrOfType<StringAttr>("lock_type").getValue();
      int lockTypeInt;
      if (lockType == "read") {
        lockTypeInt = 0;
      } else {
        lockTypeInt = 1;
      }

      auto typeConst = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), lockTypeInt));
      builtinOperands.push_back(typeConst);
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

    auto funcOp = op->template getParentOfType<ModuleOp>()
                      .template lookupSymbol<LLVM::LLVMFuncOp>(symbolName);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp, operands);
    return success();
  }
};

struct InvokeLoaderLowering
    : public AthenaConversionPattern<ath_graph::InvokeLoaderOp> {
  using AthenaConversionPattern<
      ath_graph::InvokeLoaderOp>::AthenaConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto parentFunc = op->template getParentOfType<LLVM::LLVMFuncOp>();
    auto device = parentFunc.getArgument(parentFunc.getNumArguments() - 1);
    auto allocator = parentFunc.getArgument(parentFunc.getNumArguments() - 2);

    SmallVector<Value, 3> loadArgs;
    loadArgs.push_back(allocator);
    loadArgs.push_back(device); // FIXME this must be a loader. Currently no way
                                //   to express this with Athena Graph.
    loadArgs.push_back(operands[0]);

    auto loaderOp = llvm::cast<ath_graph::InvokeLoaderOp>(op);
    auto funcOp =
        op->template getParentOfType<ModuleOp>()
            .template lookupSymbol<LLVM::LLVMFuncOp>(loaderOp.loader_routine());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp, loadArgs);

    return success();
  }
};

/// Converts builtin call to a set of instructions for launching the kernel.
///
/// To run the kernel on device, a special LaunchCommand structure must be
/// generated. The information for its fields is taken from current translation
/// unit.
///
/// The global size is currently determined by the size of return value tensor.
/// Local size is currently set to 0 for backend to decide later.
///
/// \todo actual kernel name resolution
/// \todo local size
///
/// \tparam OpT is a class of Athena Graph Dialect operation
template <typename OpT>
class BuiltinConversionPattern : public AthenaConversionPattern<OpT> {
public:
  using AthenaConversionPattern<OpT>::AthenaConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();

    auto launchCommandType = getLaunchCommandType(llvmDialect);

    llvm::errs() << OpT::getOperationName() << "\n";
    Operation* globalString =
        op->getParentOfType<ModuleOp>().lookupSymbol(OpT::getOperationName());

    // fixme fetch real kernel name
    StringRef kernelNameStr = OpT::getOperationName();

    LLVM::GlobalOp kernelNameVal;
    if (globalString) {
      kernelNameVal = llvm::cast<LLVM::GlobalOp>(globalString);
    } else {
      auto module = op->getParentOfType<ModuleOp>();
      OpBuilder builder(module);
      builder.setInsertionPointToStart(module.getBody());
      // todo string must be null-terminated.
      auto stringType = LLVM::LLVMType::getArrayTy(
          LLVM::LLVMType::getInt8Ty(llvmDialect), kernelNameStr.size());
      auto kernelNameAttr = builder.getStringAttr(kernelNameStr.data());
      kernelNameVal = builder.create<LLVM::GlobalOp>(
          builder.getUnknownLoc(), stringType, /*isConstant*/ false,
          LLVM::Linkage::Private, kernelNameStr, kernelNameAttr);
    }

    // 1. Allocate LaunchCommand structure
    auto unit = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), 1));
    auto launchCommandMem = rewriter.create<LLVM::AllocaOp>(
        op->getLoc(), launchCommandType, unit, 8);
    // 2. Set kernel name
    auto zeroArg = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));
    auto kerNameMember = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), launchCommandType.getStructElementType(0).getPointerTo(),
        launchCommandMem, ValueRange{zeroArg, zeroArg});
    auto kerNameGlobalAddr =
        rewriter.create<LLVM::AddressOfOp>(op->getLoc(), kernelNameVal);
    auto kerNameAddr = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo(),
        kerNameGlobalAddr, ValueRange{zeroArg, zeroArg});
    rewriter.create<LLVM::StoreOp>(op->getLoc(), kerNameAddr, kerNameMember);

    // 3. Set kernel args count
    auto firstArg = unit; // for consistency
    auto argCountMember = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), launchCommandType.getStructElementType(1).getPointerTo(),
        launchCommandMem, ValueRange{zeroArg, firstArg});
    auto argCountConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), getArgsCount(op)));
    rewriter.create<LLVM::StoreOp>(op->getLoc(), argCountConst, argCountMember);

    // 4. Allocate args structure
    auto argDescTy = getArgDescType(llvmDialect);
    auto argsArray = rewriter.create<LLVM::AllocaOp>(op->getLoc(), argDescTy,
                                                     argCountConst, 16);

    // 5. For each arg:
    //    i.   Set size of arg
    //    ii.  Set pointer to arg
    //    iii. Set argument type
    fillArgDesc(argsArray, op, operands, rewriter);

    auto two = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), 2));
    auto argPtr = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), getArgDescType(llvmDialect).getPointerTo(), argsArray,
        ValueRange{zeroArg, zeroArg});
    auto argMember = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), launchCommandType.getStructElementType(2).getPointerTo(),
        launchCommandMem, ValueRange{zeroArg, two});
    rewriter.create<LLVM::StoreOp>(op->getLoc(), argPtr, argMember);

    // 6. Set ND-range dimension
    auto outType = op->getOperand(op->getNumOperands() - 1).getType();
    if (!outType.isa<RankedTensorType>()) {
      llvm_unreachable("The last type must be a ranked tensor output");
    }

    auto sizetType = LLVM::LLVMType::getIntNTy(llvmDialect, sizeof(size_t) * 8);
    auto rankedTensorOut = outType.cast<RankedTensorType>();
    auto dimSize = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), sizetType,
        rewriter.getIntegerAttr(rewriter.getIntegerType(sizeof(size_t) * 8),
                                rankedTensorOut.getRank()));

    auto three = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), 3));
    auto workDimMember = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), launchCommandType.getStructElementType(3).getPointerTo(),
        launchCommandMem, ValueRange{zeroArg, three});
    rewriter.create<LLVM::StoreOp>(op->getLoc(), dimSize, workDimMember);

    // 7. Allocate mem for global size
    auto dimArrayTy =
        LLVM::LLVMType::getArrayTy(sizetType, rankedTensorOut.getRank());
    // fixme FWIW alignment must be 4 on 32-bit systems.
    auto globalSizeArray =
        rewriter.create<LLVM::AllocaOp>(op->getLoc(), dimArrayTy, dimSize, 16);

    // 8. Fill global sizes
    for (int i = 0; i < rankedTensorOut.getRank(); i++) {
      auto curIdx = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), i));

      auto curArrayElt = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), sizetType.getPointerTo(), launchCommandMem,
          ValueRange{zeroArg, curIdx});

      auto curDim = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), sizetType,
          rewriter.getIntegerAttr(rewriter.getIntegerType(sizeof(size_t) * 8),
                                  rankedTensorOut.getShape()[i]));
      rewriter.create<LLVM::StoreOp>(op->getLoc(), curDim, curArrayElt);
    }
    auto four = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), 4));
    auto globalSizePtr = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), sizetType.getPointerTo(), globalSizeArray,
        ValueRange{zeroArg, zeroArg});
    auto globalSizeMember = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), launchCommandType.getStructElementType(4).getPointerTo(),
        launchCommandMem, ValueRange{zeroArg, four});
    llvm::errs() << globalSizePtr.getType() << " " << globalSizeMember.getType()
                 << "\n";
    rewriter.create<LLVM::StoreOp>(op->getLoc(), globalSizePtr,
                                   globalSizeMember);

    // 9. Allocate memory for local size
    auto localSizeArray =
        rewriter.create<LLVM::AllocaOp>(op->getLoc(), dimArrayTy, dimSize, 16);

    // 10. Set local size to 0 (for now).
    for (int i = 0; i < rankedTensorOut.getRank(); i++) {
      auto curIdx = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), i));

      auto curArrayElt = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), sizetType.getPointerTo(), launchCommandMem,
          ValueRange{zeroArg, curIdx});

      auto zeroSize = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), sizetType,
          rewriter.getIntegerAttr(rewriter.getIntegerType(sizeof(size_t) * 8),
                                  0));
      rewriter.create<LLVM::StoreOp>(op->getLoc(), zeroSize, curArrayElt);
    }

    auto five = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), 5));
    auto localSizePtr = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), sizetType.getPointerTo(), localSizeArray,
        ValueRange{zeroArg, zeroArg});
    auto localSizeMember = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), launchCommandType.getStructElementType(5).getPointerTo(),
        launchCommandMem, ValueRange{zeroArg, five});
    rewriter.create<LLVM::StoreOp>(op->getLoc(), localSizePtr, localSizeMember);

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename FuncT>
class FunctionConversionPattern : public AthenaConversionPattern<FuncT> {
public:
  using AthenaConversionPattern<FuncT>::AthenaConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto funcOp = mlir::cast<FuncT>(op);
    auto oldType = funcOp.getType();
    SmallVector<LLVM::LLVMType, 5> newArgs;
    for (const auto& type : oldType.getInputs()) {
      Type convType = mTypeConverter.convertType(type);
      newArgs.push_back(convType.cast<LLVM::LLVMType>());
    }
    // The last two arguments have been incorrectly converted.
    newArgs.pop_back();
    newArgs.pop_back();

    auto voidPtrTy =
        LLVM::LLVMType::getInt8Ty(mTypeConverter.getDialect()).getPointerTo();

    newArgs.push_back(voidPtrTy);
    newArgs.push_back(LLVM::LLVMType::getIntNTy(
        mTypeConverter.getDialect(), mTypeConverter.getPointerWidth()));

    TypeConverter::SignatureConversion newSignature(oldType.getNumInputs());
    for (auto& en : llvm::enumerate(newArgs)) {
      newSignature.addInputs(en.index(), en.value());
    }

    LLVM::LLVMType functionReturnType =
        LLVM::LLVMType::getVoidTy(mTypeConverter.getDialect());

    // Allocator pointer
    newArgs.push_back(voidPtrTy);
    newSignature.addInputs(voidPtrTy);

    if constexpr (std::is_same_v<FuncT, ath_graph::NodeOp>) {
      // Device pointer
      newArgs.push_back(voidPtrTy);
      newSignature.addInputs(voidPtrTy);
      functionReturnType =
          LLVM::LLVMType::getInt8Ty(mTypeConverter.getDialect());
    }

    auto llvmFuncTy =
        LLVM::LLVMType::getFunctionTy(functionReturnType, newArgs, false);

    LLVM::LLVMFuncOp newFunc;
    if constexpr (std::is_same_v<FuncT, ath_graph::NodeOp>) {
      auto nodeIdAttr = rewriter.getNamedAttr(
          ath_graph::NodeOp::getNodeIdAttrName(),
          op->getAttr(ath_graph::NodeOp::getNodeIdAttrName()));
      auto clusterIdAttr = rewriter.getNamedAttr(
          ath_graph::NodeOp::getClusterIdAttrName(),
          op->getAttr(ath_graph::NodeOp::getClusterIdAttrName()));
      SmallVector<NamedAttribute, 2> attrs;
      attrs.push_back(nodeIdAttr);
      attrs.push_back(clusterIdAttr);
      newFunc = rewriter.create<LLVM::LLVMFuncOp>(
          funcOp.getLoc(), funcOp.getName(), llvmFuncTy,
          LLVM::Linkage::External, attrs);
    } else {
      newFunc = rewriter.create<LLVM::LLVMFuncOp>(funcOp.getLoc(),
                                                  funcOp.getName(), llvmFuncTy);
    }

    rewriter.inlineRegionBefore(funcOp.body(), newFunc.getBody(),
                                newFunc.getBody().end());
    rewriter.applySignatureConversion(&newFunc.getBody(), newSignature);
    rewriter.eraseOp(op);

    return success();
  }

protected:
  using AthenaConversionPattern<FuncT>::mTypeConverter;
};

class SliceLoweringPattern
    : public AthenaConversionPattern<ath_graph::SliceOp> {
public:
  using AthenaConversionPattern<ath_graph::SliceOp>::AthenaConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto funcOp =
        op->template getParentOfType<ModuleOp>()
            .template lookupSymbol<LLVM::LLVMFuncOp>("ath_get_sub_tensor");

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp, operands);
    return success();
  }
};

class EvalLoweringPattern : public AthenaConversionPattern<ath_graph::EvalOp> {
public:
  using AthenaConversionPattern<ath_graph::EvalOp>::AthenaConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto* llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    auto evalNode = llvm::cast<ath_graph::EvalOp>(op);
    auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto context = parentFunc.getArgument(parentFunc.getNumArguments() - 3);
    auto allocator = parentFunc.getArgument(parentFunc.getNumArguments() - 2);
    auto getDevFunc =
        op->getParentOfType<ModuleOp>().lookupSymbol<LLVM::LLVMFuncOp>(
            "ath_get_device_for_node");

    auto node = op->getParentOfType<ModuleOp>().lookupSymbol<LLVM::LLVMFuncOp>(
        evalNode.node());
    auto nodeIdAttr =
        node.getAttrOfType<IntegerAttr>(ath_graph::NodeOp::getNodeIdAttrName());

    auto nodeId = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt64Ty(llvmDialect),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                nodeIdAttr.getValue()));

    SmallVector<Value, 2> getDevArgs;
    getDevArgs.push_back(nodeId);
    getDevArgs.push_back(context);

    auto devPtr =
        rewriter.create<LLVM::CallOp>(op->getLoc(), getDevFunc, getDevArgs);

    SmallVector<Value, 5> nodeArgs;
    std::copy(evalNode.getArgOperands().begin(),
              evalNode.getArgOperands().end(), std::back_inserter(nodeArgs));
    nodeArgs.push_back(allocator);
    nodeArgs.push_back(devPtr.getResult(0));

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, node, nodeArgs);

    return success();
  }
};

class ReturnLoweringPattern
    : public AthenaConversionPattern<ath_graph::ReturnOp> {
public:
  using AthenaConversionPattern<ath_graph::ReturnOp>::AthenaConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
    return success();
  }

protected:
  using AthenaConversionPattern<ath_graph::ReturnOp>::mTypeConverter;
};

class GraphTerminatorLoweringPattern
    : public AthenaConversionPattern<ath_graph::GraphTerminatorOp> {
public:
  using AthenaConversionPattern<
      ath_graph::GraphTerminatorOp>::AthenaConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange{});
    return success();
  }
};

class ConstantLoweringPattern : public AthenaConversionPattern<ConstantOp> {
public:
  using AthenaConversionPattern<ConstantOp>::AthenaConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value>,
                  ConversionPatternRewriter& rewriter) const override {
    auto constOp = llvm::cast<ConstantOp>(op);

    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        op, mTypeConverter.convertType(op->getResult(0).getType()),
        constOp.getValue());
    return success();
  }

protected:
  using AthenaConversionPattern<ConstantOp>::mTypeConverter;
};

namespace {
class LowerGraphPass
    : public PassWrapper<LowerGraphPass, OperationPass<ModuleOp>> {
protected:
  void runOnOperation() override {
    OwningRewritePatternList structureLoweringPatterns;
    AthenaTypeConverter typeConverter(&getContext());
    populateGraphToRuntimeConversionPatterns(
        typeConverter, structureLoweringPatterns, &getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<ModuleTerminatorOp>();
    target.addIllegalDialect<ath_graph::AthenaGraphDialect>();
    if (failed(applyFullConversion(getOperation(), target,
                                   structureLoweringPatterns))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
void populateGraphToRuntimeConversionPatterns(
    AthenaTypeConverter& typeConverter,
    OwningRewritePatternList& loweringPatterns, MLIRContext* ctx) {
  loweringPatterns.insert<
      // clang-format off
          ReturnLoweringPattern,
          GraphTerminatorLoweringPattern,
          SliceLoweringPattern,
          InvokeLoaderLowering,
          EvalLoweringPattern,
          ConstantLoweringPattern,
          BuiltinToFuncCallLoweringPattern<ath_graph::AllocOp>,
          BuiltinToFuncCallLoweringPattern<ath_graph::ReleaseOp>,
          BuiltinToFuncCallLoweringPattern<ath_graph::GetTensor>,
          BuiltinToFuncCallLoweringPattern<ath_graph::LockOp>,
          BuiltinConversionPattern<ath_graph::AddOp>,
          BuiltinConversionPattern<ath_graph::MulOp>,
          BuiltinConversionPattern<ath_graph::MatmulOp>,
          BuiltinConversionPattern<ath_graph::TransposeOp>,
          BuiltinConversionPattern<ath_graph::FillOp>,
          FunctionConversionPattern<ath_graph::NodeOp>,
          FunctionConversionPattern<ath_graph::GraphOp>
      // clang-format on
      >(typeConverter);
}
std::unique_ptr<OperationPass<ModuleOp>> createLowerGraphToRuntimePass() {
  return std::make_unique<LowerGraphPass>();
}
} // namespace mlir
