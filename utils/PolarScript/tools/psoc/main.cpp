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

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "llvm/Support/CommandLine.h"

#include <string>

using namespace llvm;

int main(int argc, char** argv) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();
  cl::opt<std::string> OutputFilename("o", cl::desc("Specify output filename"),
                                      cl::value_desc("filename"), cl::Required);
  cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
                                     cl::desc("<input file>"), cl::init("-"));

  cl::ParseCommandLineOptions(argc, argv);

  polarai::script::CompilerFrontend frontend;

  frontend.compileFromFile(InputFilename.c_str());

  return 0;
}
