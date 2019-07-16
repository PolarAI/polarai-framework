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

#include <athena/backend/llvm/AthenaJIT.h>
#include <athena/backend/llvm/runtime-driver/runtime-driver.h>

#include <cstdlib>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>

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
        new core::FatalError(2, "Unable to detect host");
    }

    auto DL = JTMB->getDefaultDataLayoutForTarget();
    if (!DL) {
        ::llvm::consumeError(DL.takeError());
        new core::FatalError(2, "Unable to get target data layout");
    }

    return std::make_unique<AthenaJIT>(std::move(*JTMB), std::move(*DL));
}
::llvm::Error AthenaJIT::addModule(std::unique_ptr<::llvm::Module> &M) {
    return mOptimizeLayer.add(
        mExecutionSession.getMainJITDylib(),
        ::llvm::orc::ThreadSafeModule(std::move(M), mContext));
}
::llvm::Expected<::llvm::JITEvaluatedSymbol> AthenaJIT::lookup(
    ::llvm::StringRef Name) {
    return mExecutionSession.lookup({&mExecutionSession.getMainJITDylib()},
                                    mMangle(Name.str()));
}
::llvm::Expected<::llvm::orc::ThreadSafeModule> AthenaJIT::optimizeModule(
    ::llvm::orc::ThreadSafeModule TSM,
    const ::llvm::orc::MaterializationResponsibility &R) {
    ::llvm::FunctionPassManager mFunctionPassManager;
    ::llvm::ModulePassManager mModulePassManager;

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
    mModulePassManager = passBuilder.buildModuleOptimizationPipeline(
        ::llvm::PassBuilder::O2, false);
    mFunctionPassManager = passBuilder.buildFunctionSimplificationPipeline(
        ::llvm::PassBuilder::O2, ::llvm::PassBuilder::ThinLTOPhase::PostLink,
        false);

    auto lock = TSM.getContextLock();

    ::llvm::Module &module = *TSM.getModule();

    mModulePassManager.run(module, moduleAnalysisManager);

    for (auto &func : module) {
        if (!func.isDeclaration())
            mFunctionPassManager.run(func, functionAnalysisManager);
    }

    return TSM;
}

}  // namespace athena::backend::llvm
