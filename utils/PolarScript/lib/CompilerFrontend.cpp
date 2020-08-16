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

#include "CompilerFrontend.hpp"
#include "CodeGenAction.hpp"

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>

using namespace clang;
using namespace clang::driver;

namespace polarai::script {
void CompilerFrontend::compileFromFile(llvm::StringRef path) {
  std::ifstream inp(path.str());
  std::string str((std::istreambuf_iterator<char>(inp)),
                  std::istreambuf_iterator<char>());
  compileFromString(str);
}
void CompilerFrontend::compileFromString(llvm::StringRef text) {
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID(
      new clang::DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts(
      new clang::DiagnosticOptions());
  DiagOpts->ShowPresumedLoc = true;
  auto& err_ostream = llvm::errs();
  auto* DiagsPrinter =
      new clang::TextDiagnosticPrinter(err_ostream, &*DiagOpts);
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> Diags(
      new clang::DiagnosticsEngine(DiagID, &*DiagOpts, DiagsPrinter));

  auto compiler = std::make_unique<clang::CompilerInstance>();

  auto compilerInvocation = std::make_unique<clang::CompilerInvocation>();
  llvm::opt::ArgStringList CCArgs;
  CCArgs.push_back("test.cpp");
  clang::CompilerInvocation::CreateFromArgs(*compilerInvocation, CCArgs,
                                            *Diags);

  clang::CompilerInstance instance;
  instance.setInvocation(std::move(compilerInvocation));

  instance.setDiagnostics(&*Diags);
  if (!instance.hasDiagnostics())
    llvm_unreachable("Failed to create diagnostics");

  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> MemFS(
      new llvm::vfs::InMemoryFileSystem);
  OverlayFS->pushOverlay(MemFS);

  compiler->setDiagnostics(&*Diags);
  compiler->createFileManager(OverlayFS);
  compiler->createSourceManager(compiler->getFileManager());

  MemFS->addFile("test.cpp", (time_t)0,
                 llvm::MemoryBuffer::getMemBuffer(text, "test.cpp"));

  auto action = std::make_unique<CodeGenAction>();

  if (!instance.ExecuteAction(*action))
    return;
}
} // namespace polarai::script
