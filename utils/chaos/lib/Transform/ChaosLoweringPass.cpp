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

#include "ChaosLoweringPass.h"
#include "ChaosDialect.h"
#include <mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/Ops.h>

using namespace mlir;

namespace chaos {

std::unique_ptr<ChaosLoweringPass>
createChaosLoweringPass(mlir::MLIRContext* ctx) {
  return std::make_unique<ChaosLoweringPass>(ctx);
}

class ReinterpretLowering : public ConversionPattern {
private:
  LLVMTypeConverter& mTypeConverter;

public:
  ReinterpretLowering(LLVMTypeConverter& typeConverter, mlir::MLIRContext* ctx,
                      mlir::PatternBenefit benefit = 1)
      : ConversionPattern("chaos.reinterpret", benefit, ctx),
        mTypeConverter(typeConverter){};

  PatternMatchResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                  ConversionPatternRewriter& rewriter) const override {
    auto reinterpretOp = cast<ReinterpretOp>(op);
    OperandAdaptor<ReinterpretOp> transformed(operands);
    auto loc = op->getLoc();

    Type resultType;
    const auto& memRefType = reinterpretOp.getType();
    //    if () {
    auto type = memRefType.cast<MemRefType>();
    auto newType = this->mTypeConverter.convertType(type.getElementType());
    auto llvmType = newType.cast<LLVM::LLVMType>();
    resultType = llvmType.getPointerTo(type.getMemorySpace());
    //    }
    //    auto resultType =
    //    this->mTypeConverter.convertType(reinterpretOp.getType());

    auto newOp =
        rewriter.create<LLVM::BitcastOp>(loc, resultType, transformed.source());

    rewriter.replaceOp(op, newOp.getOperation()->getResults());

    return this->matchSuccess();
  }
};

void chaos::ChaosLoweringPass::runOnModule() {
  ModuleOp m = getModule();

  OwningRewritePatternList patterns;
  mlir::populateLoopToStdConversionPatterns(patterns, m.getContext());
  patterns.insert<ReinterpretLowering>(mTypeConverter, m.getContext());
  mlir::populateStdToLLVMConversionPatterns(mTypeConverter, patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();

  if (failed(applyPartialConversion(m, target, patterns, &mTypeConverter)))
    signalPassFailure();
}
} // namespace chaos