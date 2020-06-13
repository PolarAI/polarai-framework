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

#include "Conversion/RuntimeToLLVM.h"
#include "../utils/LaunchCommand.h"
#include "../utils/TensorInfo.h"
#include "ArgInfo.h"
#include "AthenaGraph/AthenaGraphDialect.h"
#include "AthenaGraph/AthenaGraphOps.h"
#include "AthenaRuntime/AthenaRuntimeDialect.h"
#include "AthenaRuntime/AthenaRuntimeOps.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/IRBuilder.h"

using namespace mlir;

static auto getVoidPtrType(LLVM::LLVMDialect* llvmDialect) -> LLVM::LLVMType {
  return LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
}

static Value allocateStructure(LLVM::LLVMType structTy,
                               ConversionPatternRewriter& rewriter,
                               Location loc) {
  auto one = rewriter.create<LLVM::ConstantOp>(
      loc, LLVM::LLVMType::getInt32Ty(&structTy.getDialect()),
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), 1));
  return rewriter.create<LLVM::AllocaOp>(loc, structTy.getPointerTo(), one, 8);
}

static Value createUInt64Constant(uint64_t value, LLVM::LLVMDialect* dialect,
                                  ConversionPatternRewriter& rewriter,
                                  Location loc) {
  return rewriter.create<LLVM::ConstantOp>(
      loc, LLVM::LLVMType::getInt64Ty(dialect),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), value));
}

static Value createUInt32Constant(uint32_t value, LLVM::LLVMDialect* dialect,
                                  ConversionPatternRewriter& rewriter,
                                  Location loc) {
  return rewriter.create<LLVM::ConstantOp>(
      loc, LLVM::LLVMType::getInt32Ty(dialect),
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), value));
}
static void setArrayEltTo(Value arrayAlloca, Value value, unsigned index,
                          ConversionPatternRewriter& rewriter, Location loc) {
  auto arrayType = arrayAlloca.getType().cast<LLVM::LLVMType>();
  auto zero = createUInt32Constant(0, &arrayType.getDialect(), rewriter, loc);
  auto idxConst =
      createUInt32Constant(index, &arrayType.getDialect(), rewriter, loc);

  auto eltPtr = rewriter.create<LLVM::GEPOp>(loc, arrayType, arrayAlloca,
                                             ValueRange{idxConst});
  rewriter.create<LLVM::StoreOp>(loc, value, eltPtr);
}

static void setStructFieldTo(Value structAlloca, LLVM::LLVMType structType,
                             Value value, unsigned index,
                             ConversionPatternRewriter& rewriter,
                             Location loc) {

  auto zero = rewriter.create<LLVM::ConstantOp>(
      loc, LLVM::LLVMType::getInt32Ty(&structType.getDialect()),
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));

  auto idxConst = rewriter.create<LLVM::ConstantOp>(
      loc, LLVM::LLVMType::getInt32Ty(&structType.getDialect()),
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), index));

  auto eltPtr = rewriter.create<LLVM::GEPOp>(
      loc, structType.getStructElementType(index).getPointerTo(), structAlloca,
      ValueRange{zero, idxConst});

  rewriter.create<LLVM::StoreOp>(loc, value, eltPtr);
}

static auto mlirTypeToDataType(mlir::Type type) -> int {
  // todo use DataType enum.
  if (type.isF64()) {
    return 1;
  } else if (type.isF32()) {
    return 2;
  } else if (type.isF16()) {
    return 3;
  }
  return 0;
}

static auto lockTypeStringToInt(llvm::StringRef str) -> uint32_t {
  if (str == "read") {
    return 0;
  } else if (str == "read_write") {
    return 1;
  }
  return -1;
}

static auto createArray(LLVM::LLVMType type, uint32_t size,
                        ConversionPatternRewriter& rewriter, Location loc)
    -> Value {
  auto sizeConst =
      createUInt32Constant(size, &type.getDialect(), rewriter, loc);
  auto arrayTy = LLVM::LLVMType::getArrayTy(type, size);
  return rewriter.create<LLVM::AllocaOp>(loc, type.getPointerTo(), sizeConst,
                                         16);
}

namespace {
template <typename OpT>
class AthenaRuntimeConversionPattern : public ConversionPattern {
public:
  AthenaRuntimeConversionPattern(LLVMTypeConverter& typeConverter,
                                 PatternBenefit patternBenefit = 1)
      : ConversionPattern(OpT::getOperationName(), patternBenefit,
                          &typeConverter.getContext()),
        mTypeConverter(typeConverter) {}

protected:
  LLVMTypeConverter& mTypeConverter;
};

/// Converts `ath_graph.create_tensor` to a set of commands that allocate and
/// fill TensorLite structure.
struct CreateTensorOpLoweringPattern
    : public AthenaRuntimeConversionPattern<ath_graph::CreateTensorOp> {
  using AthenaRuntimeConversionPattern<
      ath_graph::CreateTensorOp>::AthenaRuntimeConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto concreteOp = llvm::cast<ath_graph::CreateTensorOp>(op);
    auto tensorType = concreteOp.getType().cast<RankedTensorType>();
    auto* llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    auto tensorInfo = allocateStructure(getTensorInfoType(llvmDialect),
                                        rewriter, op->getLoc());
    auto tensorVAddr =
        createUInt64Constant(*concreteOp.virtual_address().getRawData(),
                             llvmDialect, rewriter, op->getLoc());
    setStructFieldTo(tensorInfo, getTensorInfoType(llvmDialect), tensorVAddr, 0,
                     rewriter, op->getLoc());
    auto dataType =
        createUInt32Constant(mlirTypeToDataType(tensorType.getElementType()),
                             llvmDialect, rewriter, op->getLoc());
    setStructFieldTo(tensorInfo, getTensorInfoType(llvmDialect), dataType, 1,
                     rewriter, op->getLoc());
    auto dims = createUInt64Constant(tensorType.getRank(), llvmDialect,
                                     rewriter, op->getLoc());
    setStructFieldTo(tensorInfo, getTensorInfoType(llvmDialect), dims, 2,
                     rewriter, op->getLoc());
    auto arr = createArray(LLVM::LLVMType::getInt64Ty(llvmDialect),
                           tensorType.getRank(), rewriter, op->getLoc());
    for (auto dim : llvm::enumerate(tensorType.getShape())) {
      auto dimConst = createUInt64Constant(dim.value(), llvmDialect, rewriter,
                                           op->getLoc());
      setArrayEltTo(arr, dimConst, dim.index(), rewriter, op->getLoc());
    }
    setStructFieldTo(tensorInfo, getTensorInfoType(llvmDialect), arr, 3,
                     rewriter, op->getLoc());

    rewriter.replaceOp(op, tensorInfo);

    return success();
  }
};

struct AllocOpLoweringPattern
    : AthenaRuntimeConversionPattern<ath_rt::AllocOp> {

  using AthenaRuntimeConversionPattern<
      ath_rt::AllocOp>::AthenaRuntimeConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto concreteOp = llvm::cast<ath_rt::AllocOp>(op);
    auto module = op->getParentOfType<ModuleOp>();
    auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();

    auto graphHandle = parentFunc.getArgument(0);

    auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>("ath_allocate");

    rewriter.create<LLVM::CallOp>(
        op->getLoc(), callee,
        ValueRange{graphHandle, operands[0], operands[1]});
    rewriter.eraseOp(op);

    return success();
  }
};

struct LockOpLoweringPattern : AthenaRuntimeConversionPattern<ath_rt::LockOp> {

  using AthenaRuntimeConversionPattern<
      ath_rt::LockOp>::AthenaRuntimeConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto concreteOp = llvm::cast<ath_rt::LockOp>(op);
    auto module = op->getParentOfType<ModuleOp>();
    auto* llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();

    auto graphHandle = parentFunc.getArgument(0);

    auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>("ath_lock");

    auto lockType =
        createUInt32Constant(lockTypeStringToInt(concreteOp.lock_type()),
                             llvmDialect, rewriter, op->getLoc());

    rewriter.create<LLVM::CallOp>(
        op->getLoc(), callee,
        ValueRange{graphHandle, operands[0], operands[1], lockType});
    rewriter.eraseOp(op);

    return success();
  }
};

struct ReleaseOpLoweringPattern
    : AthenaRuntimeConversionPattern<ath_rt::ReleaseOp> {

  using AthenaRuntimeConversionPattern<
      ath_rt::ReleaseOp>::AthenaRuntimeConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto concreteOp = llvm::cast<ath_rt::ReleaseOp>(op);
    auto module = op->getParentOfType<ModuleOp>();
    auto* llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();

    auto graphHandle = parentFunc.getArgument(0);

    auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>("ath_release");

    rewriter.create<LLVM::CallOp>(
        op->getLoc(), callee,
        ValueRange{graphHandle, operands[0], operands[1]});
    rewriter.eraseOp(op);

    return success();
  }
};

struct BarrierOpLoweringPattern
    : AthenaRuntimeConversionPattern<ath_rt::BarrierOp> {

  using AthenaRuntimeConversionPattern<
      ath_rt::BarrierOp>::AthenaRuntimeConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto concreteOp = llvm::cast<ath_rt::BarrierOp>(op);
    auto module = op->getParentOfType<ModuleOp>();
    auto* llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();

    auto graphHandle = parentFunc.getArgument(0);

    auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>("ath_barrier");

    auto numEvents = createUInt64Constant(concreteOp.getNumOperands(),
                                          llvmDialect, rewriter, op->getLoc());

    auto eventsArray =
        createArray(LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo(),
                    concreteOp.getNumOperands(), rewriter, op->getLoc());

    for (auto event : llvm::enumerate(operands)) {
      setArrayEltTo(eventsArray, event.value(), event.index(), rewriter,
                    op->getLoc());
    }

    rewriter.create<LLVM::CallOp>(op->getLoc(), callee,
                                  ValueRange{numEvents, eventsArray});
    rewriter.eraseOp(op);

    return success();
  }
};

struct NullEventOpLoweringPattern
    : AthenaRuntimeConversionPattern<ath_rt::NullEventOp> {

  using AthenaRuntimeConversionPattern<
      ath_rt::NullEventOp>::AthenaRuntimeConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto* llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    rewriter.replaceOpWithNewOp<LLVM::NullOp>(
        op, LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo());

    return success();
  }
};

struct DeviceSelectOpLoweringPattern
    : AthenaRuntimeConversionPattern<ath_rt::DeviceSelectOp> {

  using AthenaRuntimeConversionPattern<
      ath_rt::DeviceSelectOp>::AthenaRuntimeConversionPattern;
  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto concreteOp = llvm::cast<ath_rt::DeviceSelectOp>(op);
    auto module = op->getParentOfType<ModuleOp>();
    auto* llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();

    auto graphHandle = parentFunc.getArgument(0);

    auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>("ath_device_select");

    auto nodeId = createUInt64Constant(*concreteOp.nodeId().getRawData(),
                                       llvmDialect, rewriter, op->getLoc());

    auto res = rewriter.create<LLVM::CallOp>(op->getLoc(), callee,
                                             ValueRange{graphHandle, nodeId});
    rewriter.replaceOp(op, res.getResult(0));

    return success();
  }
};

struct InvokeLoaderOpLoweringPattern
    : AthenaRuntimeConversionPattern<ath_graph::InvokeLoaderOp> {
  using AthenaRuntimeConversionPattern<
      ath_graph::InvokeLoaderOp>::AthenaRuntimeConversionPattern;

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto concreteOp = llvm::cast<ath_graph::InvokeLoaderOp>(op);
    auto module = op->getParentOfType<ModuleOp>();
    auto* llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();

    auto graphHandle = parentFunc.getArgument(0);

    auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>("ath_load");
    auto nodeId = createUInt64Constant(
        *concreteOp
             .getAttrOfType<IntegerAttr>(ath_graph::NodeOp::getNodeIdAttrName())
             .getValue()
             .getRawData(),
        llvmDialect, rewriter, op->getLoc());

    rewriter.create<LLVM::CallOp>(op->getLoc(), callee,
                                  ValueRange{graphHandle, nodeId, operands[0]});
    rewriter.eraseOp(op);

    return success();
  }
};

struct LaunchOpLoweringPattern
    : AthenaRuntimeConversionPattern<ath_rt::LaunchOp> {
  using AthenaRuntimeConversionPattern<
      ath_rt::LaunchOp>::AthenaRuntimeConversionPattern;

  auto matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto concreteOp = llvm::cast<ath_rt::LaunchOp>(op);
    auto module = op->getParentOfType<ModuleOp>();
    auto* llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();

    concreteOp.getResult(0).replaceAllUsesWith(operands.back());

    auto argsArray = createArray(getArgDescType(llvmDialect),
                                 operands.size() - 2, rewriter, op->getLoc());

    auto argsOperands =
        llvm::iterator_range(operands.begin() + 2, operands.end());
    for (auto operand : llvm::enumerate(argsOperands)) {
      auto idx = createUInt64Constant(operand.index(), llvmDialect, rewriter,
                                      op->getLoc());
      auto argDesc = rewriter.create<LLVM::GEPOp>(op->getLoc(),
                                                  getArgDescType(llvmDialect),
                                                  argsArray, ValueRange{idx});

      auto llvmType = operand.value().getType().cast<LLVM::LLVMType>();
      if (llvmType.isPointerTy() &&
          llvmType.getPointerElementTy().getUnderlyingType()->isStructTy()) {
        // Most likely this is a tensor.
        // todo are there corner cases?

        auto tensorPtr = rewriter.create<LLVM::BitcastOp>(
            op->getLoc(), LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo(),
            operand.value());
        setStructFieldTo(argDesc, getArgDescType(llvmDialect), tensorPtr, 1,
                         rewriter, op->getLoc());
        // todo smarter way
        auto zero =
            createUInt32Constant(0, llvmDialect, rewriter, op->getLoc());
        setStructFieldTo(argDesc, getArgDescType(llvmDialect), zero, 2,
                         rewriter, op->getLoc());
      } else {
        auto sizeInBytes = createUInt64Constant(
            llvmType.getUnderlyingType()->getScalarSizeInBits() / 8,
            llvmDialect, rewriter, op->getLoc());
        setStructFieldTo(argDesc, getArgDescType(llvmDialect), sizeInBytes, 0,
                         rewriter, op->getLoc());

        auto one = createUInt64Constant(1, llvmDialect, rewriter, op->getLoc());
        auto valAlloc = rewriter.create<LLVM::AllocaOp>(
            op->getLoc(), llvmType.getPointerTo(), one, 8);
        rewriter.create<LLVM::StoreOp>(op->getLoc(), operand.value(), valAlloc);
        auto bitcastArg = rewriter.create<LLVM::BitcastOp>(
            op->getLoc(), LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo(),
            valAlloc);
        setStructFieldTo(argDesc, getArgDescType(llvmDialect), bitcastArg, 1,
                         rewriter, op->getLoc());
        auto one32 =
            createUInt32Constant(1, llvmDialect, rewriter, op->getLoc());
        setStructFieldTo(argDesc, getArgDescType(llvmDialect), one32, 2,
                         rewriter, op->getLoc());
      }
    }

    auto launchCommand = allocateStructure(getLaunchCommandType(llvmDialect),
                                           rewriter, op->getLoc());

    // Set kernel name
    auto kernelNameStr = concreteOp.kernel();

    Operation* globalString =
        op->getParentOfType<ModuleOp>().lookupSymbol(kernelNameStr);
    LLVM::GlobalOp kernelNameVal;
    if (globalString) {
      kernelNameVal = llvm::cast<LLVM::GlobalOp>(globalString);
    } else {
      OpBuilder builder(module);
      builder.setInsertionPointToStart(module.getBody());
      // todo add small string optimization for null-terminated strings.
      auto stringType = LLVM::LLVMType::getArrayTy(
          LLVM::LLVMType::getInt8Ty(llvmDialect), kernelNameStr.size() + 1);
      char* strData = new char[kernelNameStr.size() + 1];
      strcpy(strData, kernelNameStr.data());
      strData[kernelNameStr.size()] = '\00';
      auto kernelNameAttr = builder.getStringAttr(
          llvm::StringRef(strData, kernelNameStr.size() + 1));
      delete[] strData;
      kernelNameVal = builder.create<LLVM::GlobalOp>(
          builder.getUnknownLoc(), stringType, /*isConstant*/ true,
          LLVM::Linkage::Private, kernelNameStr, kernelNameAttr);
    }
    auto kerNameGlobalAddr =
        rewriter.create<LLVM::AddressOfOp>(op->getLoc(), kernelNameVal);
    auto kerNamePtr = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo(),
        kerNameGlobalAddr);
    setStructFieldTo(launchCommand, getLaunchCommandType(llvmDialect),
                     kerNamePtr, 0, rewriter, op->getLoc());

    // Set kernel arg count
    auto argCount = createUInt64Constant(operands.size() - 2, llvmDialect,
                                         rewriter, op->getLoc());
    setStructFieldTo(launchCommand, getLaunchCommandType(llvmDialect), argCount,
                     1, rewriter, op->getLoc());

    // Set args
    setStructFieldTo(launchCommand, getLaunchCommandType(llvmDialect),
                     argsArray, 2, rewriter, op->getLoc());

    // Set workDim
    auto workDim = createUInt64Constant(concreteOp.global_size().size(),
                                        llvmDialect, rewriter, op->getLoc());
    setStructFieldTo(launchCommand, getLaunchCommandType(llvmDialect), workDim,
                     3, rewriter, op->getLoc());

    // Set global size
    auto globalSize =
        createArray(LLVM::LLVMType::getInt64Ty(llvmDialect),
                    concreteOp.global_size().size(), rewriter, op->getLoc());
    for (auto s : llvm::enumerate(concreteOp.global_size())) {
      auto intAttr = s.value().cast<IntegerAttr>();

      auto size = createUInt64Constant(*intAttr.getValue().getRawData(),
                                       llvmDialect, rewriter, op->getLoc());

      setArrayEltTo(globalSize, size, s.index(), rewriter, op->getLoc());
    }
    setStructFieldTo(launchCommand, getLaunchCommandType(llvmDialect),
                     globalSize, 4, rewriter, op->getLoc());

    // Set local size
    auto localSize =
        createArray(LLVM::LLVMType::getInt64Ty(llvmDialect),
                    concreteOp.local_size().size(), rewriter, op->getLoc());
    for (auto s : llvm::enumerate(concreteOp.local_size())) {
      auto intAttr = s.value().cast<IntegerAttr>();

      auto size = createUInt64Constant(*intAttr.getValue().getRawData(),
                                       llvmDialect, rewriter, op->getLoc());

      setArrayEltTo(localSize, size, s.index(), rewriter, op->getLoc());
    }
    setStructFieldTo(launchCommand, getLaunchCommandType(llvmDialect),
                     localSize, 5, rewriter, op->getLoc());

    auto launchFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("ath_launch");

    auto graphHandle =
        concreteOp.getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);

    auto result = rewriter.create<LLVM::CallOp>(
        op->getLoc(), launchFunc,
        ValueRange{graphHandle, operands[0], operands[1], launchCommand});
    concreteOp.getResult(1).replaceAllUsesWith(result.getResult(0));
    rewriter.eraseOp(concreteOp);

    return success();
  }
};

class RuntimeToLLVM
    : public PassWrapper<RuntimeToLLVM, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    LLVMTypeConverter typeConverter(&getContext());
    auto* llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    typeConverter.addConversion([llvmDialect](ath_rt::DeviceType) {
      return getVoidPtrType(llvmDialect);
    });
    typeConverter.addConversion([llvmDialect](ath_rt::EventType) {
      return getVoidPtrType(llvmDialect);
    });
    typeConverter.addConversion([llvmDialect](ath_rt::GraphHandleType) {
      return getVoidPtrType(llvmDialect);
    });
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    populateRuntimeToLLVMConversionPatterns(typeConverter, patterns);
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<ModuleTerminatorOp>();
    if (failed(applyFullConversion(getOperation(), target, patterns))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
void populateRuntimeToLLVMConversionPatterns(
    LLVMTypeConverter& typeConverter,
    OwningRewritePatternList& loweringPatterns) {
  loweringPatterns.insert<
      // clang-format off
      CreateTensorOpLoweringPattern,
      AllocOpLoweringPattern,
      LockOpLoweringPattern,
      ReleaseOpLoweringPattern,
      BarrierOpLoweringPattern,
      NullEventOpLoweringPattern,
      DeviceSelectOpLoweringPattern,
      InvokeLoaderOpLoweringPattern,
      LaunchOpLoweringPattern
      // clang-format on
      >(typeConverter);
}
auto createLowerRuntimeToLLVMPass()
    -> std::unique_ptr<OperationPass<ModuleOp>> {
  return std::make_unique<RuntimeToLLVM>();
}
} // namespace mlir
