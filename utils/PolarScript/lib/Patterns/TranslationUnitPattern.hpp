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
class TranslationUnitPattern : public TypedPattern<clang::TranslationUnitDecl> {
public:
  TranslationUnitPattern(GenerationContext* ctx, uint64_t benefit)
      : TypedPattern<clang::TranslationUnitDecl>(ctx, benefit) {}

  llvm::StringRef getPatternName() override {
    return "TranslationUnitPattern";
  }

  MatchResult generate(clang::TranslationUnitDecl* tu,
                mlir::OpBuilder& builder) override {
    // todo support single source models
    auto gpuModule = builder.create<mlir::gpu::GPUModuleOp>(
        builder.getUnknownLoc(), "polarscript_kernels");
    builder.setInsertionPointToStart(&gpuModule.body().front());

    tu->dump();
    for (auto* decl : tu->decls()) {
      decl->dump();
      getContext()->getPatterns().generate(decl, builder);
    }

    return success();
  };
};
} // namespace polarai::script
