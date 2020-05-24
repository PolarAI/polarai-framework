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
  return rewriter.create<LLVM::AllocaOp>(loc, structTy, one, 8);
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

  auto eltPtr =
      rewriter.create<LLVM::GEPOp>(loc, arrayType.getArrayElementType(),
                                   arrayAlloca, ValueRange{zero, idxConst});
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

static Value createArray(LLVM::LLVMType type, uint32_t size,
                         ConversionPatternRewriter& rewriter, Location loc) {
  auto sizeConst =
      createUInt32Constant(size, &type.getDialect(), rewriter, loc);
  auto arrayTy = LLVM::LLVMType::getArrayTy(type, size);
  return rewriter.create<LLVM::AllocaOp>(loc, arrayTy, sizeConst, 16);
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
struct CreateTensorOpLoweringPattern
    : public AthenaRuntimeConversionPattern<ath_graph::CreateTensorOp> {
  using AthenaRuntimeConversionPattern<
      ath_graph::CreateTensorOp>::AthenaRuntimeConversionPattern;
  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) {
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
    auto arr = createArray(LLVM::LLVMType::getInt32Ty(llvmDialect),
                           tensorType.getRank(), rewriter, op->getLoc());
    for (auto dim : llvm::enumerate(tensorType.getShape())) {
      auto dimConst =
          createUInt64Constant(dim.value(), llvmDialect, rewriter, op->getLoc());
      setArrayEltTo(dims, dimConst, dim.index(), rewriter, op->getLoc());
    }
    setStructFieldTo(tensorInfo, getTensorInfoType(llvmDialect), arr, 3,
                     rewriter, op->getLoc());

    rewriter.replaceOp(op, tensorInfo);
    
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
    OwningRewritePatternList& loweringPatterns) {}
auto createLowerRuntimeToLLVMPass()
    -> std::unique_ptr<OperationPass<ModuleOp>> {
  return std::make_unique<RuntimeToLLVM>();
}
} // namespace mlir
