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

#include "MLIRGen.h"
#include "MLIRASTConsumer.h"
#include <clang/Frontend/CompilerInstance.h>
#include <mlir/IR/Builders.h>

#include <memory>

namespace chaos {
std::unique_ptr<clang::ASTConsumer>
MLIRGen::CreateASTConsumer(clang::CompilerInstance& CI,
                           llvm::StringRef InFile) {
  return std::make_unique<MLIRASTConsumer>(CI.getASTContext(), mMLIRModule);
}
MLIRGen::MLIRGen() {
  mlir::OpBuilder builder(&mMLIRContext);
  // todo module name, proper location
  mMLIRModule =
      mlir::OwningModuleRef(mlir::ModuleOp::create(builder.getUnknownLoc()));
}
} // namespace chaos