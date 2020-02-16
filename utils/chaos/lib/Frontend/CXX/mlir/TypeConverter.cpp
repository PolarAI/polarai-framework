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

#include "TypeConverter.h"
#include <Dialects/ClangDialect.h>

#include <clang/AST/PrettyPrinter.h>
#include <clang/Basic/LangOptions.h>
#include <mlir/Dialect/StandardOps/Ops.h>

namespace chaos {

mlir::FunctionType
chaos::TypeConverter::convert(const clang::FunctionType& type) {
  if (type.isFunctionNoProtoType()) {
    llvm_unreachable("Strange things have happened");
  }

  auto protoType = type.getAs<clang::FunctionProtoType>();
  auto paramTypes = protoType->getParamTypes();
  llvm::SmallVector<mlir::Type, 5> params;
  for (auto& param : paramTypes) {
    params.push_back(convert(param));
  }

  // Workaround for void type. MLIR's None type is actually
  // nothing like C++ void. So, void functions must return
  // 0 result values.
  if (protoType->getReturnType().getAsString() == "void") {
    return mBuilder.getFunctionType(params, {});
  }

  auto retType = convert(protoType->getReturnType());

  return mBuilder.getFunctionType(params, {retType});
}

mlir::Type chaos::TypeConverter::convert(const clang::QualType& type) {
  mlir::Type resType;

  if (type->isRecordType()) {
    auto name = mangleTypeName(*type->getAsStructureType());
    if (mRegisteredTypes.count(name)) {
      return mRegisteredTypes[name];
    }
    llvm::errs() << name << "\n";
    llvm_unreachable("Typename not found");
  } else if (type->isAnyPointerType()) {
    auto pointerTy = type->getAs<clang::PointerType>();
    auto pointee = convert(pointerTy->getPointeeType());
    if (pointee.isa<mlir::NoneType>()) { // workaround for "void*"
      resType = clang::RawPointerType::get(pointee);
    } else {
      resType = mlir::MemRefType::get({-1}, pointee);
    }
  } else if (type->isLValueReferenceType()) {
    auto refType = type->getAs<clang::LValueReferenceType>();
    auto pointee = convert(refType->getPointeeType());
    resType = clang::RawPointerType::get(pointee);
  } else if (type->isRValueReferenceType()) {
    auto refType = type->getAs<clang::RValueReferenceType>();
    auto pointee = convert(refType->getPointeeType());
    resType = clang::RawPointerType::get(pointee);
  } else if (type->isBuiltinType()) {
    return convert(*type->getAs<clang::BuiltinType>());
  } else {
    llvm::errs() << "Unknown type: " << type->getTypeClassName();
    llvm_unreachable(".");
  }

  return resType;
} // namespace chaos
mlir::Type chaos::TypeConverter::convert(const clang::BuiltinType& type) {
  mlir::Type resType;

  auto name =
      std::string(type.getName(clang::PrintingPolicy(clang::LangOptions())));
  if (type.isFloatingType()) {
    if (name == "float") {
      resType = mBuilder.getF32Type();
    } else if (name == "double") {
      resType = mBuilder.getF64Type();
    }
  } else if (type.isBooleanType()) {
    resType = mBuilder.getI1Type();
  } else if (type.isVoidType()) {
    resType = mBuilder.getNoneType();
  } else {
    if (name == "int" || name == "unsigned int" || name == "const int") {
      resType = mBuilder.getIntegerType(32);
    } else if (name == "long" || name == "unsigned long") {
      // todo windows must be 32
      resType = mBuilder.getIntegerType(64);
    } else if (name == "long long") {
      resType = mBuilder.getIntegerType(64);
    }
  }

  return resType;
}
mlir::Type TypeConverter::getAsPointer(const clang::QualType& type) {
  auto mlirType = convert(type);
  return mlir::MemRefType::get({-1}, mlirType);
}
void TypeConverter::registerType(llvm::StringRef name, mlir::Type type) {
  mRegisteredTypes.insert({name.str(), type});
}
std::string TypeConverter::mangleTypeName(const clang::RecordType& type) {
  std::string name;
  llvm::raw_string_ostream outStream(name);
  mMangleContext.mangleCXXName(type.getDecl(), outStream);
  outStream.flush();
  return name;
}
std::string TypeConverter::mangleTypeName(const clang::QualType& type) {
  std::string name;
  llvm::raw_string_ostream outStream(name);
  mMangleContext.mangleTypeName(type, outStream);
  outStream.flush();
  return name;
}
} // namespace chaos
