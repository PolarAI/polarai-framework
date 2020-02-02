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

#include "PythonFrontend.h"
#include "Lexer.h"
#include "Parser.h"

#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

namespace chaos {
void PythonFrontend::run(const std::vector<std::string>& args) {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(args[0]);

  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return;
  }

  auto buffer = fileOrErr.get()->getBuffer();
  BufLexer lexer(buffer.begin(), buffer.end(), std::string(args[0]));

  Parser parser(lexer);

  auto ast = parser.buildAST();
  ast.dump();
}
} // namespace chaos