//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "clang/AST/Type.h"

#include <vector>

namespace polarai::script {
class TypeConversionPattern {
public:
  int getBenefit() { return mBenefit; }

private:
  int mBenefit;
};
class TypeConverter {
public:
  mlir::Type convert(const clang::QualType& type, mlir::OpBuilder& builder) {
    if (type->isPointerType()) {
      auto scalarType = type->getPointeeType();
      auto mlirType = convert(scalarType, builder);
      // todo support address spaces.
      return mlir::MemRefType::get({-1}, mlirType);
    } else if (type->isBuiltinType()) {
      auto builtinT = type->getAs<clang::BuiltinType>();
      if (builtinT->isInteger()) {
        return builder.getI32Type();
      }
    }

    return mlir::Type{};
  }

private:
  std::vector<TypeConversionPattern> mPatterns;
};
} // namespace polarai::script
