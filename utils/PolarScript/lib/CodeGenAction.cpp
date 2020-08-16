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

#include "CodeGenAction.hpp"
#include "MlirGen.hpp"

#include "clang/Frontend/CompilerInstance.h"

namespace polarai::script {
std::unique_ptr<clang::ASTConsumer>
CodeGenAction::CreateASTConsumer(clang::CompilerInstance& Compiler,
                                 llvm::StringRef InFile) {
  mlir::OpBuilder builder(&mMLIRContext);
  mModule =
      mlir::OwningModuleRef(mlir::ModuleOp::create(builder.getUnknownLoc()));

  return std::make_unique<MLIRGen>(&Compiler.getASTContext(), mModule);
}
} // namespace polarai::script
