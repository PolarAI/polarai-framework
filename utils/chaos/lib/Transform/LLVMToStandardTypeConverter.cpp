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

#include "LLVMToStandardTypeConverter.h"

using namespace mlir;

mlir::Type chaos::LLVMToStandardTypeConverter::convertType(mlir::Type t) {
  return TypeConverter::convertType(t);
}
mlir::Type chaos::LLVMToStandardTypeConverter::convertIntegerType(
    mlir::LLVM::LLVMType type) {
  if (type.getUnderlyingType()->isIntegerTy()) {
    return IntegerType::get(type.getUnderlyingType()->getIntegerBitWidth(),
                            type.getContext());
  }
  llvm_unreachable("Not an integer type");
}
mlir::Type chaos::LLVMToStandardTypeConverter::convertFloatType(
    mlir::LLVM::LLVMType type) {

  if (type.getUnderlyingType()->isDoubleTy()) {
    return FloatType::getF64(type.getContext());
  } else if (type.getUnderlyingType()->isFloatTy()) {
    return FloatType::getF32(type.getContext());
  } else if (type.getUnderlyingType()->isHalfTy()) {
    return FloatType::getF16(type.getContext());
  }
  llvm_unreachable("Not a supported FP type");
}
mlir::Type chaos::LLVMToStandardTypeConverter::convertPointerType(
    mlir::LLVM::LLVMType type) {
  if (type.getUnderlyingType()->isPointerTy()) {
    return MemRefType::get({-1},
                           convertStandardType(type.getPointerElementTy()), {},
                           type.getUnderlyingType()->getPointerAddressSpace());
  }
  llvm_unreachable("Unsupported type");
}
mlir::FunctionType chaos::LLVMToStandardTypeConverter::convertFunctionType(
    mlir::LLVM::LLVMType type) {
  SignatureConversion conversion(type.getFunctionNumParams());
  FunctionType converted =
      convertFunctionSignature(type, /*isVariadic=*/false, conversion);
  return converted;
}
// todo handle variadic parameters
mlir::FunctionType chaos::LLVMToStandardTypeConverter::convertFunctionSignature(
    mlir::LLVM::LLVMType type, bool isVariadic,
    TypeConverter::SignatureConversion& result) {
  SmallVector<Type, 1> returnTypes;
  auto retType = convertStandardType(type.getFunctionResultType());
  if (!retType.isa<NoneType>()) {
    returnTypes.push_back(retType);
  }

  for (size_t i = 0; i < type.getFunctionNumParams(); ++i) {
    auto argType = type.getFunctionParamType(i);
    auto newType = convertStandardType(argType);
    result.addInputs(i, newType);
  }

  SmallVector<Type, 8> argTypes;
  argTypes.reserve(llvm::size(result.getConvertedTypes()));
  for (Type convertedType : result.getConvertedTypes())
    argTypes.push_back(convertedType);

  return FunctionType::get(argTypes, returnTypes, retType.getContext());
}
mlir::Type chaos::LLVMToStandardTypeConverter::convertStandardType(
    mlir::LLVM::LLVMType type) {
  if (type.isIntegerTy()) {
    return convertIntegerType(type);
  } else if (type.getUnderlyingType()->isFloatingPointTy()) {
    return convertFloatType(type);
  } else if (type.getUnderlyingType()->isVoidTy()) {
    return NoneType::get(type.getContext());
  } else if (type.isPointerTy()) {
    return convertPointerType(type);
  } else if (type.isFunctionTy()) {
    return convertFunctionType(type);
  }
  llvm_unreachable("Unsupported type ");
}
