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

#include <Target/ObjectEmitter.h>
#include <llvm/CodeGen/CommandFlags.inc>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/LTO/LTO.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetOptions.h>

#include <utility>

using namespace llvm;

static void check(Error err, const std::string& msg) {
  if (!err)
    return;
  handleAllErrors(std::move(err), [&](ErrorInfoBase& EIB) {
    errs() << msg << ": " << EIB.message().c_str() << '\n';
  });
  exit(1);
}

template <typename T> static T check(Expected<T> err, const std::string& msg) {
  if (err)
    return std::move(*err);
  check(err.takeError(), msg);
  return T();
}

static void check(std::error_code errCode, const std::string& msg) {
  check(errorCodeToError(errCode), std::move(msg));
}

template <typename T> static T check(ErrorOr<T> err, const std::string& msg) {
  if (err)
    return std::move(*err);
  check(err.getError(), msg);
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

std::vector<std::string>
ObjectEmitter::emitObject(const std::string& filename) {
  lto::Config config;
  config.DiagHandler = [](const DiagnosticInfo& diagnosticInfo) {
    DiagnosticPrinterRawOStream oStream(errs());
    diagnosticInfo.print(oStream);
    errs() << '\n';
    if (diagnosticInfo.getSeverity() == DS_Error)
      exit(1);
  };
  config.UseNewPM = true;
  config.OptLevel = 2;
  config.CodeGenOnly = false;
  config.CGOptLevel = CodeGenOpt::Default;
  config.DefaultTriple = ::llvm::sys::getDefaultTargetTriple();
  config.OverrideTriple = ::llvm::sys::getDefaultTargetTriple();
  if (auto RM = getRelocModel())
    config.RelocModel = *RM;
  config.CodeModel = getCodeModel();
  config.Options = InitTargetOptionsFromCodeGenFlags();
  config.Options.FunctionSections = true;
  config.CPU = getCPUStr();

  check(config.addSaveTemps(filename + "."), "Config::addSaveTemps failed");

  lto::ThinBackend ltoBackend = lto::createInProcessThinBackend(1);
  lto::LTO lto(std::move(config), std::move(ltoBackend), 1);

  for (auto& inpBuf : mLinkableModules) {
    std::unique_ptr<lto::InputFile> input =
        check(lto::InputFile::create(inpBuf->getMemBufferRef()),
              "Failed to create input");
    auto syms = input->symbols();
    std::vector<lto::SymbolResolution> resols(syms.size());

    for (size_t i = 0; i < syms.size(); i++) {
      lto::SymbolResolution& r = resols[i];

      r.Prevailing = !syms[i].isUndefined();
      r.VisibleToRegularObj = !syms[i].canBeOmittedFromSymbolTable();
      r.FinalDefinitionInLinkageUnit = true;
      r.LinkerRedefined = true;
    }

    check(lto.add(std::move(input), resols), "");
  }

  std::vector<std::string> res;
  auto addStream =
      [&](size_t taskId) -> std::unique_ptr<lto::NativeObjectStream> {
    std::string path = filename + "." + utostr(taskId);
    res.push_back(path);

    std::error_code errorCode;
    auto stream =
        std::make_unique<raw_fd_ostream>(path, errorCode, sys::fs::OF_None);
    check(errorCode, path);
    return std::make_unique<lto::NativeObjectStream>(std::move(stream));
  };

  lto::NativeObjectCache cache;

  check(lto.run(addStream, cache), "LTO::run failed");

  return res;
}
