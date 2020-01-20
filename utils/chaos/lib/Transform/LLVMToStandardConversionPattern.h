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

#ifndef ATHENA_LLVMTOSTANDARDCONVERSIONPATTERN_H
#define ATHENA_LLVMTOSTANDARDCONVERSIONPATTERN_H

#include "LLVMToStandardTypeConverter.h"
#include <mlir/Transforms/DialectConversion.h>

namespace chaos {
class LLVMToStandardConversionPattern : public mlir::ConversionPattern {
public:
  LLVMToStandardConversionPattern(llvm::StringRef opName,
                                  mlir::MLIRContext* ctx,
                                  mlir::PatternBenefit benefit = 1)
      : ConversionPattern(opName, benefit, ctx) {}
};

template <typename SrcOp>
class LLVMToStdLoweringPattern : public LLVMToStandardConversionPattern {
protected:
  LLVMToStandardTypeConverter& mTypeConverter;
  mlir::StandardOpsDialect& mDialect;

public:
  LLVMToStdLoweringPattern(mlir::StandardOpsDialect& dialect,
                           LLVMToStandardTypeConverter& converter)
      : LLVMToStandardConversionPattern(SrcOp::getOperationName(),
                                        dialect.getContext()),
        mTypeConverter(converter), mDialect(dialect) {}
};

void registerStdConversionPatterns(LLVMToStandardTypeConverter& converter,
                                   mlir::OwningRewritePatternList& patterns);

} // namespace chaos

#endif // ATHENA_LLVMTOSTANDARDCONVERSIONPATTERN_H
