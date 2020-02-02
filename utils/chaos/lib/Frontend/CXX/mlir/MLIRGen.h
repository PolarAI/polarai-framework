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

#ifndef ATHENA_MLIRGEN_H
#define ATHENA_MLIRGEN_H

#include <clang/Frontend/FrontendAction.h>
#include <mlir/IR/Module.h>

namespace chaos {
class MLIRGen : public clang::ASTFrontendAction {
protected:
  mlir::MLIRContext mMLIRContext;
  mlir::OwningModuleRef mMLIRModule;

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance& CI,
                    llvm::StringRef InFile) override;

public:
  MLIRGen();
  mlir::OwningModuleRef& getModule() { return mMLIRModule; }
  mlir::OwningModuleRef takeModule() { return std::move(mMLIRModule); }
};
} // namespace chaos

#endif // ATHENA_MLIRGEN_H
