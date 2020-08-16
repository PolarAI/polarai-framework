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

#include "Pattern.hpp"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "clang/AST/Decl.h"

namespace polarai::script {
class TypedefPattern : public TypedPattern<clang::TypedefDecl> {
public:
  TypedefPattern(GenerationContext* ctx, uint64_t benefit)
      : TypedPattern<clang::TypedefDecl>(ctx, benefit) {}

  llvm::StringRef getPatternName() override {
    return "TypedefPattern";
  }

  MatchResult generate(clang::TypedefDecl* decl, mlir::OpBuilder& builder) override {
    decl->dump();
    // fixme implement typedef alias declarations
    return success();
  };
};
} // namespace polarai::script

