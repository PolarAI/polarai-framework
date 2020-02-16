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

#include <Driver/DriverOptions.h>

#include <Dialects/ClangDialect.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/FrontendTool/Utils.h>
#include <llvm/Option/Option.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/IR/Dialect.h>

#include "mlir/MLIRGen.h"

namespace chaos {
chaos::CXXFrontend::CXXFrontend(std::shared_ptr<DriverOptions> opts)
    : mOptions(std::move(opts)), mDiagnosticID(new clang::DiagnosticIDs()),
      mDiagnosticOpts(new clang::DiagnosticOptions()),
      mDiagnosticPrinter(new clang::TextDiagnosticPrinter(
          llvm::errs(), mDiagnosticOpts.get())),
      mDiagnosticsEngine(new clang::DiagnosticsEngine(
          mDiagnosticID, mDiagnosticOpts.get(), mDiagnosticPrinter)),
      mCompilerInstance(new clang::CompilerInstance()) {
  mDiagnosticOpts->ShowPresumedLoc = true;

  llvm::InitializeAllTargets();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  mlir::registerDialect<clang::ClangDialect>();
}
void CXXFrontend::run(const std::string& fileName) {
  auto compilerInvocation = std::make_unique<clang::CompilerInvocation>();

  auto fullOpts = getCXXFlags(fileName);
  llvm::SmallVector<const char*, 16> argsRef;
  for (auto& opt : fullOpts) {
    argsRef.push_back(opt.data());
  }

  clang::CompilerInvocation::CreateFromArgs(*compilerInvocation, argsRef,
                                            *mDiagnosticsEngine);

  mCompilerInstance->createDiagnostics();
  if (!mCompilerInstance->hasDiagnostics())
    return;

  mCompilerInstance->setInvocation(std::move(compilerInvocation));

  if (mOptions->UseMlir) {
    std::unique_ptr<MLIRGen> Act(new MLIRGen());
    if (!mCompilerInstance->ExecuteAction(*Act)) {
      llvm::errs() << "Frontend Action failed\n";
      return;
    }
    if (mOptions->DumpMlir) {
      Act->getModule()->print(llvm::outs());
    }
  }
}
std::vector<std::string> CXXFrontend::getCXXFlags(const std::string& fileName) {
  std::array<char, 128> buffer{};
  std::string path;

  std::unique_ptr<FILE, decltype(&pclose)> pipe(
      popen("llvm-config --prefix", "r"), pclose);

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    path += buffer.data();
  }

  llvm::Triple triple(mOptions->TargetTriple.getValue());

  clang::driver::Driver driver(path, triple.str(), *mDiagnosticsEngine);
  driver.setTitle("chaos");
  driver.setCheckInputsExist(false);

  // fixme output path
  llvm::SmallVector<const char*, 4> allArgs{"-fsyntax-only", fileName.data()};

#ifdef __APPLE__
  allArgs.push_back("-isysroot");
  allArgs.push_back("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk");
#endif

  std::unique_ptr<clang::driver::Compilation> clang(
      driver.BuildCompilation(allArgs));

  const clang::driver::JobList& jobs = clang->getJobs();

  const clang::driver::Command& command =
      llvm::cast<clang::driver::Command>(*jobs.begin());

  std::vector<std::string> res;
  auto args = command.getArguments();
  for (const auto* arg : args) {
    res.emplace_back(arg);
  }

  return res;
}
} // namespace chaos
