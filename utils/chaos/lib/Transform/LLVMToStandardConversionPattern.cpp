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

#include "LLVMToStandardConversionPattern.h"
#include "ChaosDialect.h"

using namespace mlir;

namespace chaos {
class LLVMFuncConversion : public LLVMToStdLoweringPattern<LLVM::LLVMFuncOp> {
public:
  using LLVMToStdLoweringPattern<LLVM::LLVMFuncOp>::LLVMToStdLoweringPattern;

  PatternMatchResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto funcOp = cast<LLVM::LLVMFuncOp>(op);
    auto funcType = funcOp.getType();

    TypeConverter::SignatureConversion result(funcOp.getNumArguments());
    auto newFuncType =
        mTypeConverter.convertFunctionSignature(funcType, false, result);

    SmallVector<NamedAttribute, 4> attributes;
    for (const auto& attr : funcOp.getAttrs()) {
      if (attr.first.is(SymbolTable::getSymbolAttrName()) ||
          attr.first.is(impl::getTypeAttrName()) ||
          attr.first.is("std.varargs"))
        continue;
      attributes.push_back(attr);
    }

    auto newFuncOp = rewriter.create<FuncOp>(funcOp.getLoc(), funcOp.getName(),
                                             newFuncType, attributes);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);

    rewriter.eraseOp(op);

    return matchSuccess();
  };
};

template <typename SourceOp, typename TargetOp>
class OneToOneOpConverter : public LLVMToStdLoweringPattern<SourceOp> {
public:
  using LLVMToStdLoweringPattern<SourceOp>::LLVMToStdLoweringPattern;
  using Super = OneToOneOpConverter<SourceOp, TargetOp>;

  PatternMatchResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto loc = op->getLoc();

    SmallVector<Type, 3> resultTypes;

    for (auto type : op->getResultTypes()) {
      if (type.isa<LLVM::LLVMType>()) {
        auto llvmType = type.cast<LLVM::LLVMType>();
        resultTypes.push_back(
            this->mTypeConverter.convertStandardType(llvmType));
      } else {
        resultTypes.push_back(type);
      }
    }

    auto newOp = rewriter.create<TargetOp>(loc, resultTypes, op->getOperands(),
                                           op->getAttrList().getAttrs());

    rewriter.replaceOp(op, newOp.getOperation()->getResults());

    return this->matchSuccess();
  }
};

struct ICmpConverter : public OneToOneOpConverter<LLVM::ICmpOp, CmpIOp> {
  using Super::Super;
};

struct ConstConverter
    : public OneToOneOpConverter<LLVM::ConstantOp, ConstantOp> {
  using Super::Super;
};

template <typename SourceOp, typename TargetOp>
struct LoadStoreConverter : public LLVMToStdLoweringPattern<SourceOp> {
public:
  using LLVMToStdLoweringPattern<SourceOp>::LLVMToStdLoweringPattern;
  using Super = LoadStoreConverter<SourceOp, TargetOp>;

  PatternMatchResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto loc = op->getLoc();

    SmallVector<Type, 3> resultTypes;

    for (auto type : op->getResultTypes()) {
      if (type.isa<LLVM::LLVMType>()) {
        auto llvmType = type.cast<LLVM::LLVMType>();
        resultTypes.push_back(
            this->mTypeConverter.convertStandardType(llvmType));
      } else {
        resultTypes.push_back(type);
      }
    }

    SmallVector<Value, 3> newOperands;
    size_t idx = 0;

    if constexpr (std::is_same_v<SourceOp, LLVM::StoreOp>) {
      idx = 1;
      newOperands.push_back(operands[0]);
    }

    SmallVector<Value, 3> results;
    if (operands[idx].getDefiningOp()->getName().getStringRef() ==
        "llvm.getelementptr") {
      for (const auto& o : operands[idx].getDefiningOp()->getOperands()) {
        bool isPointer = o.getType().isa<MemRefType>();

        if (o.getType().isa<LLVM::LLVMType>()) {
          auto llvmType = o.getType().cast<LLVM::LLVMType>();
          isPointer |= llvmType.isPointerTy();
        }

        if (!isPointer) {
          OpBuilder builder(op);
          auto index = builder.create<IndexCastOp>(
              loc, o, IndexType::get(op->getContext()));
          newOperands.push_back(index);
        } else {
          newOperands.push_back(o);
        }
      }
    } else {
      newOperands.push_back(operands[idx]);
    }

    auto newOp = rewriter.create<TargetOp>(loc, resultTypes, newOperands,
                                           op->getAttrList().getAttrs());

    for (const auto& res : newOp.getOperation()->getResults()) {
      results.push_back(res);
    }
    rewriter.replaceOp(op, results);

    return this->matchSuccess();
  }
};

struct GEPConverter : public LLVMToStdLoweringPattern<LLVM::GEPOp> {
public:
  using LLVMToStdLoweringPattern<LLVM::GEPOp>::LLVMToStdLoweringPattern;

  PatternMatchResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return this->matchSuccess();
  }
};

struct LoadConverter : public LoadStoreConverter<LLVM::LoadOp, LoadOp> {
  using Super::Super;
};

struct StoreConverter : public LoadStoreConverter<LLVM::StoreOp, StoreOp> {
  using Super::Super;
};

struct BitcastConverter
    : public OneToOneOpConverter<LLVM::BitcastOp, chaos::ReinterpretOp> {
  using Super::Super;
};

struct CallConverter : public OneToOneOpConverter<LLVM::CallOp, CallOp> {
  using Super::Super;
};

static void rewiriteBlockTypes(ArrayRef<Block*> blocks,
                               LLVMToStandardTypeConverter& converter) {
  for (auto* block : blocks) {
    for (auto& arg : block->getArguments()) {
      if (arg.getType().isa<LLVM::LLVMType>()) {
        arg.setType(converter.convertStandardType(
            arg.getType().cast<LLVM::LLVMType>()));
      }
    }
  }
}

struct CondBrConverter : public LLVMToStdLoweringPattern<LLVM::CondBrOp> {
  using LLVMToStdLoweringPattern<LLVM::CondBrOp>::LLVMToStdLoweringPattern;

  PatternMatchResult
  matchAndRewrite(Operation* op, ArrayRef<Value> properOperands,
                  ArrayRef<Block*> destinations,
                  ArrayRef<ArrayRef<Value>> operands,
                  ConversionPatternRewriter& rewriter) const override {
    rewiriteBlockTypes(destinations, this->mTypeConverter);
    SmallVector<ValueRange, 2> operandRanges(operands.begin(), operands.end());
    rewriter.replaceOpWithNewOp<CondBranchOp>(
        op, properOperands[0], destinations[0], operandRanges[0],
        destinations[1], operandRanges[1]);
    return this->matchSuccess();
  }
};

struct BrConverter : public LLVMToStdLoweringPattern<LLVM::BrOp> {
  using LLVMToStdLoweringPattern<LLVM::BrOp>::LLVMToStdLoweringPattern;

  PatternMatchResult
  matchAndRewrite(Operation* op, ArrayRef<Value> properOperands,
                  ArrayRef<Block*> destinations,
                  ArrayRef<ArrayRef<Value>> operands,
                  ConversionPatternRewriter& rewriter) const override {
    rewiriteBlockTypes(destinations, this->mTypeConverter);
    SmallVector<ValueRange, 2> operandRanges(operands.begin(), operands.end());
    rewriter.replaceOpWithNewOp<BranchOp>(op, destinations[0],
                                          operandRanges[0]);
    return this->matchSuccess();
  }
};

struct ReturnConverter : public LLVMToStdLoweringPattern<LLVM::ReturnOp> {
  using LLVMToStdLoweringPattern<LLVM::ReturnOp>::LLVMToStdLoweringPattern;

  PatternMatchResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    SmallVector<Value, 1> newOperands(operands.begin(), operands.end());
    rewriter.replaceOpWithNewOp<ReturnOp>(op, newOperands);
    return this->matchSuccess();
  }
};

template <typename SourceOp, typename TargetOp>
using BinaryOpConverter = OneToOneOpConverter<SourceOp, TargetOp>;

struct FAddOpConverter : public BinaryOpConverter<LLVM::FAddOp, AddFOp> {
  using Super::Super;
};
struct AddOpConverter : public BinaryOpConverter<LLVM::AddOp, AddIOp> {
  using Super::Super;
};
struct FMulOpConverter : public BinaryOpConverter<LLVM::FMulOp, MulFOp> {
  using Super::Super;
};
struct MulOpConverter : public BinaryOpConverter<LLVM::MulOp, MulIOp> {
  using Super::Super;
};

void registerStdConversionPatterns(LLVMToStandardTypeConverter& converter,
                                   mlir::OwningRewritePatternList& patterns) {
  // Function
  patterns.insert<LLVMFuncConversion>(*converter.getDialect(), converter);

  // Arithmetic ops
  patterns
      .insert<AddOpConverter, FAddOpConverter, MulOpConverter, FMulOpConverter>(
          *converter.getDialect(), converter);

  // Utility ops
  patterns.insert<ICmpConverter, CondBrConverter, ConstConverter, BrConverter,
                  ReturnConverter, CallConverter>(*converter.getDialect(),
                                                  converter);

  // Memory ops
  patterns.insert<LoadConverter, StoreConverter, BitcastConverter>(
      *converter.getDialect(), converter);

  patterns.insert<GEPConverter>(*converter.getDialect(), converter);
}
} // namespace chaos