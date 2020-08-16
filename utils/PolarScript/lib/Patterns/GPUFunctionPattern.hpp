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

#include "GenerationContext.hpp"
#include "Pattern.hpp"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "clang/AST/Decl.h"

namespace polarai::script {
class GPUFunctionPattern : public TypedPattern<clang::FunctionDecl> {
public:
  GPUFunctionPattern(GenerationContext* ctx, uint64_t benefit)
      : TypedPattern<clang::FunctionDecl>(ctx, benefit) {}

  auto getPatternName() -> llvm::StringRef override {
    return "GPUFunctionPattern";
  }

  auto generate(clang::FunctionDecl* func, mlir::OpBuilder& builder)
      -> MatchResult override {
    llvm::SmallVector<mlir::Type, 5> funcParamTypes;

    for (auto* param : func->parameters()) {
      funcParamTypes.push_back(getContext()->getTypeConverter().convert(
          param->getOriginalType(), builder));
    }

    auto funcType =
        mlir::FunctionType::get(funcParamTypes, {}, builder.getContext());
    auto mlirFunc = builder.create<mlir::gpu::GPUFuncOp>(
        builder.getUnknownLoc(), func->getName(), funcType);

    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> ParamsScope(
        getContext()->getSymbolTable());

    for (auto& param : llvm::enumerate(mlirFunc.getArguments())) {
      getContext()->getSymbolTable().insert(
          func->getParamDecl(param.index())->getName(), param.value());
    }

    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(&mlirFunc.getBody().front());
    getContext()->getPatterns().generate(func->getBody(), builder);

    return success();
  };
};
} // namespace polarai::script
