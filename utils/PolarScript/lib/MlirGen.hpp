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

#include "Pattern.hpp"
#include "GenerationContext.hpp"

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Mangle.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Module.h>

namespace polarai::script {

constexpr uint64_t CommonBenefit = 0;
constexpr uint64_t LanguageBenefit = 1;
constexpr uint64_t ToolBenefit = 2;

void populateStandalonePatterns(PatternList& list, GenerationContext*,
                                uint64_t benefit);

class MLIRGen : public clang::ASTConsumer {
public:
  MLIRGen(clang::ASTContext* context, mlir::OwningModuleRef& module)
      : mContext(context), mMLIRModule(module), mBuilder(module.get()),
        mMangleContext(std::unique_ptr<clang::MangleContext>(
            clang::ItaniumMangleContext::create(*mContext,
                                                mContext->getDiagnostics()))) {
    mGenCtx = std::make_unique<GenerationContext>(mContext, mMangleContext);

    mBuilder.setInsertionPointToStart(mMLIRModule->getBody());

    populateStandalonePatterns(mGenCtx->getPatterns(), mGenCtx.get(),
                               CommonBenefit);
  }
  void HandleTranslationUnit(clang::ASTContext& context) override;

  ~MLIRGen() override = default;

private:
  clang::ASTContext* mContext;
  mlir::OwningModuleRef& mMLIRModule;
  mlir::OpBuilder mBuilder;
  std::unique_ptr<clang::MangleContext> mMangleContext;
  std::unique_ptr<GenerationContext> mGenCtx;
};
} // namespace polarai::script
