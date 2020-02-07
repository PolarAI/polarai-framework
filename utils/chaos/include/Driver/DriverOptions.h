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

#ifndef ATHENA_DRIVEROPTIONS_H
#define ATHENA_DRIVEROPTIONS_H

#include <llvm/Support/CommandLine.h>

#include <memory>

namespace chaos {

struct DriverOptions {
  llvm::cl::opt<std::string> OutputFilename{
      llvm::cl::opt<std::string>("o", llvm::cl::desc("Specify output filename"),
                                 llvm::cl::value_desc("filename"))};
  llvm::cl::list<std::string> InputFilenames{
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required,
      llvm::cl::OneOrMore};
  llvm::cl::opt<std::string> TargetTriple{
      "triple", llvm::cl::desc("Target triple"), llvm::cl::value_desc("triple"),
      llvm::cl::Optional, llvm::cl::init(llvm::sys::getProcessTriple())};
  llvm::cl::opt<bool> FrontendOnly{"frontend-only",
                                   llvm::cl::desc("Only run the frontend")};
  llvm::cl::opt<bool> UseMlir{
      "use-mlir",
      llvm::cl::desc("Use MLIR dialect as intermediate representation")};
  llvm::cl::opt<bool> DumpMlir{"dump-mlir",
                               llvm::cl::desc("Print MLIR to stdout")};
};
} // namespace chaos

#endif // ATHENA_DRIVEROPTIONS_H
