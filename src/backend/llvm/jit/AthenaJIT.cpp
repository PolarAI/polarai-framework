/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "AthenaJIT.h"

#include <athena/core/FatalError.h>

#include <cstdlib>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/PartialInlining.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>

static size_t getNextFileId() {
    static size_t count = 0;
    return ++count;
}

namespace athena::backend::llvm {
AthenaJIT::AthenaJIT(::llvm::orc::JITTargetMachineBuilder JTMB,
                     ::llvm::DataLayout &&DL)
    : mObjectLayer(
          mExecutionSession,
          []() { return ::llvm::make_unique<::llvm::SectionMemoryManager>(); }),
      mCompileLayer(mExecutionSession,
                    mObjectLayer,
                    ::llvm::orc::ConcurrentIRCompiler(std::move(JTMB))),
      mOptimizeLayer(mExecutionSession, mCompileLayer, optimizeModule),
      mMergeLayer(mExecutionSession, mOptimizeLayer),
      mDataLayout(DL),
      mMangle(mExecutionSession, mDataLayout),
      mContext(::llvm::make_unique<::llvm::LLVMContext>()) {
    ::llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
    mExecutionSession.getMainJITDylib().setGenerator(cantFail(
        ::llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(DL)));
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
::llvm::Error AthenaJIT::addModule(std::unique_ptr<::llvm::Module> &M) {
    return mMergeLayer.add(
        mExecutionSession.getMainJITDylib(),
        ::llvm::orc::ThreadSafeModule(std::move(M), mContext),
        mExecutionSession.allocateVModule());
}
::llvm::Expected<::llvm::JITEvaluatedSymbol> AthenaJIT::lookup(
    ::llvm::StringRef Name) {
    return mExecutionSession.lookup({&mExecutionSession.getMainJITDylib()},
                                    mMangle(Name.str()));
}
::llvm::Expected<::llvm::orc::ThreadSafeModule> AthenaJIT::optimizeModule(
    ::llvm::orc::ThreadSafeModule TSM,
    const ::llvm::orc::MaterializationResponsibility &R) {
#ifdef DEBUG
    size_t fileId = getNextFileId();
    std::error_code errorCode;
    const std::string fileNamePrefix = "program" + std::to_string(fileId);

    if (getenv("ATHENA_DUMP_LLVM")) {
        auto lock = TSM.getContextLock();
        ::llvm::raw_fd_ostream preOptStream(fileNamePrefix + "_pre_opt.ll",
                                            errorCode);
        TSM.getModule()->print(preOptStream, nullptr);
        preOptStream.close();
    }
#endif

    ::llvm::FunctionPassManager mFunctionSimplificationPassManager;
    ::llvm::ModulePassManager mModuleOptimizationPassManager;
    ::llvm::ModulePassManager mThinLTOPreLinkPassManager;
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
    passBuilder.crossRegisterProxies(
        loopAnalysisManager, functionAnalysisManager, cGSCCAnalysisManager,
        moduleAnalysisManager);
    mModuleOptimizationPassManager =
        passBuilder.buildModuleOptimizationPipeline(::llvm::PassBuilder::O2,
                                                    false);
    mFunctionSimplificationPassManager =
        passBuilder.buildFunctionSimplificationPipeline(
            ::llvm::PassBuilder::O2,
            ::llvm::PassBuilder::ThinLTOPhase::PostLink, false);
    mThinLTOPreLinkPassManager =
        passBuilder.buildThinLTOPreLinkDefaultPipeline(::llvm::PassBuilder::O3);

    mDefaultIPOPassManager.addPass(::llvm::AlwaysInlinerPass());
    mDefaultIPOPassManager.addPass(::llvm::PartialInlinerPass());

    auto lock = TSM.getContextLock();

    ::llvm::Module &module = *TSM.getModule();

    mModuleOptimizationPassManager.run(module, moduleAnalysisManager);
    mThinLTOPreLinkPassManager.run(module, moduleAnalysisManager);
    mDefaultIPOPassManager.run(module, moduleAnalysisManager);

    for (auto &func : module) {
        if (!func.isDeclaration())
            mFunctionSimplificationPassManager.run(func,
                                                   functionAnalysisManager);
    }

#ifdef DEBUG
    if (getenv("ATHENA_DUMP_LLVM")) {
        ::llvm::raw_fd_ostream postOptStream(fileNamePrefix + "_post_opt.ll",
                                             errorCode);
        TSM.getModule()->print(postOptStream, nullptr);
    }
#endif

    return TSM;
}

}  // namespace athena::backend::llvm
