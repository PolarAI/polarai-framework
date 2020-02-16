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

#include <Dialects/ClangDialect.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/StandardTypes.h>

clang::ClangDialect::ClangDialect(mlir::MLIRContext* ctx)
    : mlir::Dialect("clang", ctx) {
  addOperations<
#define GET_OP_LIST
#include <Dialects/ClangDialect.cpp.inc>
      >();
  addTypes<StructType, RawPointerType>();
}

namespace clang {
using namespace llvm;
#define GET_OP_CLASSES
#include <Dialects/ClangDialect.cpp.inc>

namespace detail {
struct StructTypeStorage : public mlir::TypeStorage {
  llvm::ArrayRef<mlir::Type> mMemberTypes;
  using KeyTy = llvm::ArrayRef<mlir::Type>;
  StructTypeStorage(llvm::ArrayRef<mlir::Type> memberTypes)
      : mMemberTypes(memberTypes) {}
  bool operator==(const KeyTy& key) const { return key == mMemberTypes; }
  static llvm::hash_code hashKey(const KeyTy& key) {
    return llvm::hash_value(key);
  }
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> memberTypes) {
    return KeyTy(memberTypes);
  }
  static StructTypeStorage* construct(mlir::TypeStorageAllocator& allocator,
                                      const KeyTy& key) {
    llvm::ArrayRef<mlir::Type> memberTypes = allocator.copyInto(key);
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(memberTypes);
  }
};
struct RawPointerTypeStorage : public mlir::TypeStorage {
  mlir::Type mPointeeType;
  using KeyTy = mlir::Type;
  RawPointerTypeStorage(mlir::Type pointee) : mPointeeType(pointee) {}
  bool operator==(const KeyTy& key) const { return key == mPointeeType; }
  static llvm::hash_code hashKey(const KeyTy& key) {
    return llvm::hash_value(key.getKind());
  }
  static KeyTy getKey(mlir::Type pointee) { return KeyTy(pointee); }
  static RawPointerTypeStorage* construct(mlir::TypeStorageAllocator& allocator,
                                          const KeyTy& key) {
    return new (allocator.allocate<RawPointerTypeStorage>())
        RawPointerTypeStorage(key);
  }
};
} // namespace detail

StructType StructType::get(llvm::ArrayRef<mlir::Type> memberTypes) {
  mlir::MLIRContext* ctx = memberTypes.front().getContext();
  return Base::get(ctx, ClangMLIRTypes::Struct, memberTypes);
}
llvm::ArrayRef<mlir::Type> StructType::getMemberTypes() {
  return getImpl()->mMemberTypes;
}

RawPointerType RawPointerType::get(mlir::Type pointee) {
  mlir::MLIRContext* ctx = pointee.getContext();
  return Base::get(ctx, ClangMLIRTypes::Pointer, pointee);
}
mlir::Type RawPointerType::getPointeeType() { return getImpl()->mPointeeType; }

void StructDeclOp::build(Builder* builder, OperationState& result,
                         llvm::StringRef name, StructType type) {
  result.addAttribute("structName", builder->getStringAttr(name));
  result.addAttribute("structType", TypeAttr::get(type));
}

mlir::Type ClangDialect::parseType(mlir::DialectAsmParser& parser) const {
  // Parse a struct type in the following form:
  // struct-type ::= `struct` `<` type (`,` type)* `>`
  if (!parser.parseKeyword("struct")) { // parse `struct`
    if (parser.parseLess())             // parse `<`
      return Type();

    // Parse the member types of the struct.
    SmallVector<mlir::Type, 3> memberTypes;
    do {
      // Parse the current member type.
      mlir::Type memberType;
      if (parser.parseType(memberType))
        return nullptr;

      memberTypes.push_back(memberType);

      // Parse the optional `,`
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseGreater()) // parse `>`
      return Type();
    return StructType::get(memberTypes);
  } else if (!parser.parseKeyword("ptr")) {
    if (parser.parseLess()) // parse `<`
      return Type();
    mlir::Type pointee;
    if (parser.parseType(pointee))
      return nullptr;
    if (parser.parseGreater()) // parse `>`
      return Type();
    return RawPointerType::get(pointee);
  }
  return Type();
}
void clang::ClangDialect::printType(mlir::Type type,
                                    mlir::DialectAsmPrinter& printer) const {
  if (type.isa<StructType>()) {
    auto structType = type.cast<StructType>();
    printer << "struct<";
    mlir::interleaveComma(structType.getMemberTypes(), printer);
    printer << '>';
  } else if (type.isa<RawPointerType>()) {
    auto ptrType = type.cast<RawPointerType>();
    printer << "ptr<" << ptrType.getPointeeType() << ">";
  }
}
} // namespace clang