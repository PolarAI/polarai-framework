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

#ifndef ATHENA_TYPECONVERTER_H
#define ATHENA_TYPECONVERTER_H

#include <clang/AST/Mangle.h>
#include <clang/AST/Type.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Types.h>

#include <unordered_map>

namespace chaos {
class TypeConverter {
  mlir::OpBuilder mBuilder;
  llvm::StringMap<mlir::Type> mRegisteredTypes;
  clang::MangleContext& mMangleContext;

public:
  TypeConverter(mlir::MLIRContext* ctx, clang::MangleContext& mangleContext)
      : mBuilder(ctx), mMangleContext(mangleContext) {}

  mlir::FunctionType convert(const clang::FunctionType& type);
  mlir::Type convert(const clang::QualType& type);
  mlir::Type convert(const clang::BuiltinType& type);
  mlir::Type getAsPointer(const clang::QualType& type);

  void registerType(llvm::StringRef name, mlir::Type type);

  std::string mangleTypeName(const clang::RecordType& type);
  std::string mangleTypeName(const clang::QualType& type);
};
} // namespace chaos

#endif // ATHENA_TYPECONVERTER_H
