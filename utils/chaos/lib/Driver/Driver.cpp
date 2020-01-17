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

#include <Driver/Driver.h>
//#include <Target/ObjectEmitter.h>
#include <Transform/IRTransformer.h>
#include <array>
#include <cstdio>
#include <iostream>
#include <llvm/Support/CommandLine.h>
#include <memory>

namespace chaos {

using namespace llvm;

static bool hasEnding(std::string const& fullString,
                      std::string const& ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(),
                                    ending.length(), ending));
  } else {
    return false;
  }
}

void Driver::run(int argc, char** argv) {
  cl::ResetCommandLineParser();
  cl::opt<std::string> OutputFilename("o", cl::desc("Specify output filename"),
                                      cl::value_desc("filename"));
  cl::list<std::string> InputFilenames(cl::Positional, cl::desc("<input file>"),
                                       cl::Required, cl::OneOrMore);
  cl::ParseCommandLineOptions(argc, argv);

  std::vector<std::string> cppInput;
  std::string outputFile = OutputFilename.getValue();

  // todo libclang
  for (auto& inp : InputFilenames) {
    if (hasEnding(inp, ".cpp")) {
      cppInput.push_back(inp);
    }
  }

  std::vector<std::string> rawLLVMIR;

  size_t idx = 0;
  for (auto& cpp : cppInput) {
    // todo better random name generator
    std::string tmp = "/tmp/chaos" + std::to_string(idx++) + ".ll";
    std::string cmd = "clang++ -std=c++17 -S -emit-llvm -fno-exceptions "
                      "-fno-rtti -mllvm -disable-llvm-optzns ";
    cmd += "-o " + tmp + " " + cpp;
    rawLLVMIR.push_back(tmp);
    std::cerr << exec(cmd);
  }

  std::vector<std::string> optimizedBitcode;

  for (auto& llvmIr : rawLLVMIR) {
    std::string tmp = "/tmp/chaos" + std::to_string(idx++) + ".bc";
    auto transformer = IRTransformer::getFromIrFile(llvmIr);
    transformer->run();
    transformer->writeBitcode(tmp);
    optimizedBitcode.push_back(tmp);
  }

  // todo(abatashev): investigate ThinLTO crashes
  //
  //  ObjectEmitter emitter;
  //  for (auto& module : optimizedBitcode) {
  //    emitter.addModule(module);
  //  }
  //
  //  emitter.emitObject(OutputFilename.getValue());

  std::string linkCmd = "clang++ -o " + OutputFilename.getValue() + " ";
  for (auto& bc : optimizedBitcode) {
    linkCmd += bc + " ";
  }
  std::cerr << "Link: " << exec(linkCmd);
}
std::string Driver::exec(const std::string& cmd) {
  std::array<char, 128> buffer{};
  std::string result;

  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                pclose);

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }

  return result;
}
} // namespace chaos
