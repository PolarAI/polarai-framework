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

#include "clang/Frontend/FrontendAction.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"

namespace polarai::script {
class CodeGenAction : public clang::ASTFrontendAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance& Compiler,
                    llvm::StringRef InFile) override;

  void EndSourceFileAction() {
    // todo remove before commit
    mModule->dump();
  }
private:
  mlir::MLIRContext mMLIRContext;
  mlir::OwningModuleRef mModule;
};
} // namespace polarai::script
