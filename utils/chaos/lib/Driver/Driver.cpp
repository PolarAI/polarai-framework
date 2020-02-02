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

#include "clang/Driver/Driver.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Job.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include <Driver/Driver.h>
#include <Frontend/Frontend.h>
#include <Target/ObjectEmitter.h>
#include <Transform/IRTransformer.h>
#include <array>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <cstdio>
#include <iostream>
#include <llvm/Support/CommandLine.h>
#include <memory>

static int kBinaryAddr;

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
  std::vector<std::string> objectFiles;
  std::string outputFile = OutputFilename.getValue();

  // todo libclang
  for (auto& inp : InputFilenames) {
    if (hasEnding(inp, ".cpp")) {
      cppInput.push_back(inp);
    } else if (hasEnding(inp, ".o")) {
      objectFiles.push_back(inp);
    }
  }

  std::vector<std::string> rawLLVMIR;

  Frontend frontend;
  auto cxxFlags = getCXXFlags(argv[0]);

  size_t idx = 0;
  for (auto& cpp : cppInput) {
    // todo better random name generator
    std::string tmp = "/tmp/chaos" + std::to_string(idx++) + ".ll";
    std::string cmd = "clang++ -std=c++17 -S -emit-llvm -fno-exceptions "
                      "-fno-rtti -mllvm -disable-llvm-optzns ";
    cmd += "-o " + tmp + " " + cpp;
    rawLLVMIR.push_back(tmp);
    std::cerr << exec(cmd);
    frontend.run(cpp, cxxFlags);
  }

  std::vector<std::string> optimizedBitcode;

  for (auto& llvmIr : rawLLVMIR) {
    std::string tmp = "/tmp/chaos" + std::to_string(idx++) + ".bc";
    auto transformer = IRTransformer::getFromIrFile(llvmIr);
    transformer->run();
    transformer->writeBitcode(tmp);
    transformer->dumpMLIR("test.mlir");
    optimizedBitcode.push_back(tmp);
  }

  ObjectEmitter emitter;
  for (auto& module : optimizedBitcode) {
    emitter.addModule(module);
  }

  std::string tmpObjectFile = "/tmp/chaos" + std::to_string(idx) + ".o";
  auto tmpObjects = emitter.emitObject(tmpObjectFile);
  objectFiles.insert(objectFiles.end(), tmpObjects.begin(), tmpObjects.end());

  std::string linkCmd = "clang++ -o " + OutputFilename.getValue() + " ";
  for (auto& o : objectFiles) {
    linkCmd += o + " ";
  }
  std::cerr << exec(linkCmd);
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
std::vector<std::string> Driver::getCXXFlags(const char* thisBin) {
  void* MainAddr = (void*)(intptr_t)kBinaryAddr;
  // std::string Path = llvm::sys::fs::getMainExecutable(thisBin, MainAddr);
  std::string Path = "/opt/llvm10/bin/clang";
  IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts =
      new clang::DiagnosticOptions();
  auto* DiagClient = new clang::TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

  IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID(new clang::DiagnosticIDs());
  clang::DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);

  const std::string TripleStr = llvm::sys::getProcessTriple();
  llvm::Triple T(TripleStr);

  clang::driver::Driver TheDriver(Path, T.str(), Diags);
  TheDriver.setTitle("clang interpreter");
  TheDriver.setCheckInputsExist(false);

  SmallVector<const char*, 16> Args{"-fsyntax-only", "-c", "test.cpp"};

  Args.push_back("-isysroot");
  Args.push_back("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk");

  std::unique_ptr<clang::driver::Compilation> C(
      TheDriver.BuildCompilation(Args));

  const clang::driver::JobList& Jobs = C->getJobs();

  //  if (Jobs.size() != 1 || !isa<clang::driver::Command>(*Jobs.begin())) {
  //    SmallString<256> Msg;
  //    llvm::raw_svector_ostream OS(Msg);
  //    Jobs.Print(OS, "; ", true);
  //    Diags.Report(clang::diag::err_fe_expected_compiler_job) << OS.str();
  //  }

  const clang::driver::Command& Cmd =
      cast<clang::driver::Command>(*Jobs.begin());

  std::vector<std::string> res;
  auto args = Cmd.getArguments();
  for (const auto* arg : args) {
    res.emplace_back(arg);
  }

  Jobs.Print(llvm::errs(), "\n", true);
  llvm::errs() << "\n";

  return res;
}
} // namespace chaos
