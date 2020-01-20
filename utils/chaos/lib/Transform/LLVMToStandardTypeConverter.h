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

#ifndef ATHENA_LLVMTOSTANDARDTYPECONVERTER_H
#define ATHENA_LLVMTOSTANDARDTYPECONVERTER_H

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/Ops.h>
#include <mlir/Transforms/DialectConversion.h>

namespace chaos {
class LLVMToStandardTypeConverter : public mlir::TypeConverter {
protected:
  mlir::StandardOpsDialect* mDialect;

public:
  LLVMToStandardTypeConverter(mlir::StandardOpsDialect* dialect)
      : mDialect(dialect){};

  using mlir::TypeConverter::convertType;

  mlir::Type convertType(mlir::Type t) override;

  mlir::FunctionType convertFunctionSignature(mlir::LLVM::LLVMType type,
                                              bool isVariadic,
                                              SignatureConversion& result);

  mlir::StandardOpsDialect* getDialect() { return mDialect; }

  mlir::Type convertStandardType(mlir::LLVM::LLVMType type);

private:
  mlir::FunctionType convertFunctionType(mlir::LLVM::LLVMType type);

  mlir::Type convertIntegerType(mlir::LLVM::LLVMType type);

  mlir::Type convertFloatType(mlir::LLVM::LLVMType type);

  mlir::Type convertPointerType(mlir::LLVM::LLVMType type);
};
} // namespace chaos

#endif // ATHENA_LLVMTOSTANDARDTYPECONVERTER_H
