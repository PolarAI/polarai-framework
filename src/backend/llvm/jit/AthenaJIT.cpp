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

#include "AthenaJIT.h"

#include <athena/core/FatalError.h>

#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/PartialInlining.h>
#include <llvm/Transforms/Scalar.h>

#include <cstdlib>

using namespace llvm::orc;

static size_t getNextFileId() {
  static size_t count = 0;
  return ++count;
}

static void explodeOnLazyCompileFailure() {
  llvm::errs() << "Lazy compilation failed, Symbol Implmentation not found!\n";
  exit(1);
}

namespace athena::backend::llvm {
AthenaJIT::AthenaJIT(::llvm::orc::JITTargetMachineBuilder JTMB,
                     ::llvm::DataLayout&& DL)
    : mDataLayout(DL), mMangle(mExecutionSession, mDataLayout),
      mMainJD(mExecutionSession.createJITDylib("<main>")),
      mContext(std::make_unique<::llvm::LLVMContext>()) {
  ::llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);

  auto callThroughMgr = createLocalLazyCallThroughManager(
      JTMB.getTargetTriple(), mExecutionSession,
      ::llvm::pointerToJITTargetAddress(explodeOnLazyCompileFailure));

  if (!callThroughMgr) {
    new core::FatalError(core::ATH_FATAL_OTHER, "");
  }

  mCallThroughManager = std::move(*callThroughMgr);

  mObjectLayer =
      std::make_unique<RTDyldObjectLinkingLayer>(mExecutionSession, []() {
        return std::make_unique<::llvm::SectionMemoryManager>();
      });
#if (LLVM_VERSION_MAJOR == 11)
  auto irCompiler = std::make_unique<ConcurrentIRCompiler>(JTMB);
#else
  ConcurrentIRCompiler irCompiler(JTMB);
#endif
  mCompileLayer = std::make_unique<IRCompileLayer>(
      mExecutionSession, *mObjectLayer, std::move(irCompiler));
  mOptimizeLayer = std::make_unique<IRTransformLayer>(
      mExecutionSession, *mCompileLayer, optimizeModule);

  mCODLayer = std::make_unique<CompileOnDemandLayer>(
      mExecutionSession, *mOptimizeLayer, *mCallThroughManager,
      createLocalIndirectStubsManagerBuilder(JTMB.getTargetTriple()));
  mCODLayer->setImplMap(&mSymbolMap);

  mExecutionSession.setDispatchMaterialization(
      [this](JITDylib& jitDylib,
             std::unique_ptr<MaterializationUnit> materializationUnit) {
        auto matUnitPtr = std::shared_ptr<MaterializationUnit>(
            std::move(materializationUnit));
        auto Work = [matUnitPtr{std::move(matUnitPtr)}, &jitDylib]() {
          matUnitPtr->doMaterialize(jitDylib);
        };
        mCompileThreads.async(std::move(Work));
      });
  mCODLayer->setPartitionFunction(CompileOnDemandLayer::compileWholeModule);

  setUpJITDylib(&mMainJD);
}
std::unique_ptr<AthenaJIT> AthenaJIT::create() {
  LLVMInitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();
  LLVMInitializeNativeAsmParser();

  auto JTMB = ::llvm::orc::JITTargetMachineBuilder::detectHost();

  if (!JTMB) {
    ::llvm::consumeError(JTMB.takeError());
    new core::FatalError(core::ATH_FATAL_OTHER, "Unable to detect host");
  }

  auto DL = JTMB->getDefaultDataLayoutForTarget();
  if (!DL) {
    ::llvm::consumeError(DL.takeError());
    new core::FatalError(core::ATH_FATAL_OTHER,
                         "Unable to get target data layout");
  }

  return std::make_unique<AthenaJIT>(std::move(*JTMB), std::move(*DL));
}
::llvm::Error AthenaJIT::addModule(std::unique_ptr<::llvm::Module>& M) {
  return mCODLayer->add(mMainJD,
                        ::llvm::orc::ThreadSafeModule(std::move(M), mContext));
}
::llvm::Expected<::llvm::JITEvaluatedSymbol>
AthenaJIT::lookup(::llvm::StringRef name) {
  return mExecutionSession.lookup({&mMainJD}, mMangle(name.str()));
}
::llvm::Expected<::llvm::orc::ThreadSafeModule>
AthenaJIT::optimizeModule(::llvm::orc::ThreadSafeModule TSM,
                          const ::llvm::orc::MaterializationResponsibility&
                              materializationResponsibility) {
#ifdef DEBUG
  size_t fileId = getNextFileId();
  std::error_code errorCode;
  const std::string fileNamePrefix = "program" + std::to_string(fileId);

  if (getenv("ATHENA_DUMP_LLVM")) {
    TSM.withModuleDo([&](::llvm::Module& module) {
      ::llvm::raw_fd_ostream preOptStream(fileNamePrefix + "_pre_opt.ll",
                                          errorCode);
      if (!errorCode) {
        module.print(preOptStream, nullptr);
        preOptStream.close();
      } else {
        log() << "Unable to open file for writing "
              << fileNamePrefix + "_pre_opt.ll";
      }
    });
  }
#endif

  ::llvm::FunctionPassManager mFunctionSimplificationPassManager;
  ::llvm::ModulePassManager mModuleOptimizationPassManager;
  ::llvm::ModulePassManager mDefaultIPOPassManager;

  ::llvm::LoopAnalysisManager loopAnalysisManager;
  ::llvm::FunctionAnalysisManager functionAnalysisManager;
  ::llvm::CGSCCAnalysisManager cGSCCAnalysisManager;
  ::llvm::ModuleAnalysisManager moduleAnalysisManager;

  ::llvm::PassBuilder passBuilder;
  passBuilder.registerModuleAnalyses(moduleAnalysisManager);
  passBuilder.registerCGSCCAnalyses(cGSCCAnalysisManager);
  passBuilder.registerFunctionAnalyses(functionAnalysisManager);
  passBuilder.registerLoopAnalyses(loopAnalysisManager);
  passBuilder.crossRegisterProxies(loopAnalysisManager, functionAnalysisManager,
                                   cGSCCAnalysisManager, moduleAnalysisManager);
  mModuleOptimizationPassManager = passBuilder.buildModuleOptimizationPipeline(
      ::llvm::PassBuilder::OptimizationLevel::O2, false);
  mFunctionSimplificationPassManager =
      passBuilder.buildFunctionSimplificationPipeline(
          ::llvm::PassBuilder::OptimizationLevel::O2,
          ::llvm::PassBuilder::ThinLTOPhase::PostLink, false);

  mDefaultIPOPassManager.addPass(::llvm::AlwaysInlinerPass());
  mDefaultIPOPassManager.addPass(::llvm::PartialInlinerPass());

  TSM.withModuleDo([&](::llvm::Module& module) {
    mModuleOptimizationPassManager.run(module, moduleAnalysisManager);
    mDefaultIPOPassManager.run(module, moduleAnalysisManager);

    for (auto& func : module) {
      if (!func.isDeclaration())
        mFunctionSimplificationPassManager.run(func, functionAnalysisManager);
    }
  });

#ifdef DEBUG
  if (getenv("ATHENA_DUMP_LLVM")) {
    TSM.withModuleDo([&](::llvm::Module& module) {
      ::llvm::raw_fd_ostream postOptStream(fileNamePrefix + "_post_opt.ll",
                                           errorCode);
      if (!errorCode) {
        module.print(postOptStream, nullptr);
        postOptStream.close();
      } else {
        log() << "Unable to open file for writing "
              << fileNamePrefix + "_post_opt.ll";
      }
    });
  }
#endif

  return TSM;
}
AthenaJIT::~AthenaJIT() { mCompileThreads.wait(); }
void AthenaJIT::setUpJITDylib(JITDylib* jitDylib) {
  LocalCXXRuntimeOverrides cxxRuntimeOverrides;

  auto err = cxxRuntimeOverrides.enable(*jitDylib, mMangle);
  if (err) {
    // todo print err message
    new core::FatalError(core::ATH_FATAL_OTHER, "Unexpected JIT error");
  }

  char prefix = mDataLayout.getGlobalPrefix();
  auto generator = DynamicLibrarySearchGenerator::GetForCurrentProcess(prefix);
  if (!generator) {
    new core::FatalError(core::ATH_FATAL_OTHER, "Failed to create generator");
  }
  jitDylib->addGenerator(std::move(*generator));
}

} // namespace athena::backend::llvm
