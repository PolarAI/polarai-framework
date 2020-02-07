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
#ifndef ATHENA_CXXFRONTEND_H
#define ATHENA_CXXFRONTEND_H

#include <Driver/DriverOptions.h>

#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/Option/Option.h>

namespace chaos {
class CXXFrontend {
private:
  std::shared_ptr<DriverOptions> mOptions;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> mDiagnosticID;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> mDiagnosticOpts;
  // fixme destructor crash
  clang::TextDiagnosticPrinter* mDiagnosticPrinter;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> mDiagnosticsEngine;
  std::unique_ptr<clang::CompilerInstance> mCompilerInstance;

  std::vector<std::string> getCXXFlags(const std::string& fileName);

public:
  CXXFrontend(std::shared_ptr<DriverOptions> opts);

  void run(const std::string& fileName);
};
} // namespace chaos
#endif // ATHENA_CXXFRONTEND_H
