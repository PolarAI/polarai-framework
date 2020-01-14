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

#include "llvm/CodeGen/CommandFlags.inc"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Threading.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <Target/ObjectEmitter.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/LTO/Caching.h>
#include <llvm/LTO/LTO.h>
#include <llvm/Support/SourceMgr.h>

#include <utility>

using namespace llvm;

static void check(Error E, std::string Msg) {
  if (!E)
    return;
  handleAllErrors(std::move(E), [&](ErrorInfoBase& EIB) {
    errs() << "llvm-lto2: " << Msg << ": " << EIB.message().c_str() << '\n';
  });
  exit(1);
}

template <typename T> static T check(Expected<T> E, std::string Msg) {
  if (E)
    return std::move(*E);
  check(E.takeError(), Msg);
  return T();
}

static void check(std::error_code EC, std::string Msg) {
  check(errorCodeToError(EC), std::move(Msg));
}

template <typename T> static T check(ErrorOr<T> E, std::string Msg) {
  if (E)
    return std::move(*E);
  check(E.getError(), Msg);
  return T();
}

ObjectEmitter::ObjectEmitter() {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();
  InitializeAllTargetInfos();
}

void ObjectEmitter::addModule(const std::string& filename) {
  SMDiagnostic err;
  mLinkableModules.push_back(
      std::move(check(MemoryBuffer::getFile(filename), "")));
}

void ObjectEmitter::emitObject(const std::string& filename) {
  lto::Config Conf;
  Conf.DiagHandler = [](const DiagnosticInfo& DI) {
    DiagnosticPrinterRawOStream DP(errs());
    DI.print(DP);
    errs() << '\n';
    if (DI.getSeverity() == DS_Error)
      exit(1);
  };
  //  Conf.CPU = "generic";
  //  Conf.Options = InitTargetOptionsFromCodeGenFlags();
  Conf.UseNewPM = true;
  Conf.OptLevel = 2;
  Conf.CodeGenOnly = false;
  Conf.DebugPassManager = true;
  Conf.CGOptLevel = CodeGenOpt::Default;
  Conf.DefaultTriple = ::llvm::sys::getDefaultTargetTriple();
  Conf.OverrideTriple = ::llvm::sys::getDefaultTargetTriple();
  if (auto RM = getRelocModel())
    Conf.RelocModel = *RM;
  Conf.CodeModel = getCodeModel();
  Conf.Options = InitTargetOptionsFromCodeGenFlags();
  Conf.Options.FunctionSections = true;
  Conf.DisableVerify = false;
  Conf.CPU = getCPUStr();

  check(Conf.addSaveTemps(filename + "."), "Config::addSaveTemps failed");

  lto::ThinBackend Backend = lto::createInProcessThinBackend(1);
  lto::LTO lto(std::move(Conf), std::move(Backend), 1);

  for (auto& inpBuf : mLinkableModules) {
    std::unique_ptr<lto::InputFile> Input =
        check(lto::InputFile::create(inpBuf->getMemBufferRef()),
              "Failed to create input");
    auto syms = Input->symbols();
    std::vector<lto::SymbolResolution> resols(syms.size());

    for (size_t i = 0; i < syms.size(); i++) {
      lto::SymbolResolution& r = resols[i];

      r.Prevailing = !syms[i].isUndefined();
      r.VisibleToRegularObj = !syms[i].canBeOmittedFromSymbolTable();
      r.FinalDefinitionInLinkageUnit = true;
      r.LinkerRedefined = true;
    }

    check(lto.add(std::move(Input), resols), "");
  }

  auto AddStream =
      [&](size_t Task) -> std::unique_ptr<lto::NativeObjectStream> {
    std::string Path = filename + "." + utostr(Task);

    std::error_code EC;
    auto S = std::make_unique<raw_fd_ostream>(Path, EC, sys::fs::OF_None);
    check(EC, Path);
    return std::make_unique<lto::NativeObjectStream>(std::move(S));
  };

  lto::NativeObjectCache Cache;

  check(lto.run(AddStream, Cache), "LTO::run failed");
}
