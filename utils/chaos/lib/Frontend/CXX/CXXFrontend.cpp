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

#include "CXXFrontend.h"
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/FrontendTool/Utils.h>
#include <llvm/Option/Option.h>
#include <llvm/Support/TargetSelect.h>

#include "mlir/MLIRGen.h"

namespace chaos {
chaos::CXXFrontend::CXXFrontend()
    : mCompilerInstance(new clang::CompilerInstance()) {
  llvm::InitializeAllTargets();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
}
void CXXFrontend::run(const std::vector<std::string>& args) {
  std::unique_ptr<clang::CompilerInvocation> CI(new clang::CompilerInvocation);

  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID(
      new clang::DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts(
      new clang::DiagnosticOptions());
  DiagOpts->ShowPresumedLoc = true;
  auto* DiagsPrinter =
      new clang::TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> Diags(
      new clang::DiagnosticsEngine(DiagID, &*DiagOpts, DiagsPrinter));

  llvm::SmallVector<const char*, 16> argsRef;
  for (auto& arg : args) {
    argsRef.push_back(arg.c_str());
  }

  clang::CompilerInvocation::CreateFromArgs(*CI, argsRef, *Diags);

  mCompilerInstance->createDiagnostics();
  if (!mCompilerInstance->hasDiagnostics())
    return;

  mCompilerInstance->setInvocation(std::move(CI));

  std::unique_ptr<MLIRGen> Act(new MLIRGen());
  std::unique_ptr<clang::ASTFrontendAction> print(new clang::ASTDumpAction());
  if (!mCompilerInstance->ExecuteAction(*Act)) {
    llvm::errs() << "Frontend Action failed\n";
    return;
  }
}
} // namespace chaos
