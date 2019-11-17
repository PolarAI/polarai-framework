/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "MergeLayer.h"

#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace athena::backend::llvm {

void MergeLayer::emit(::llvm::orc::MaterializationResponsibility responsibility,
                      ::llvm::orc::ThreadSafeModule threadSafeModule) {
    assert(threadSafeModule && "Module must not be null");
    mMaterializedMap[responsibility.getTargetJITDylib().getName()] = true;
    mBaseLayer.emit(std::move(responsibility), std::move(threadSafeModule));
}

::llvm::Error MergeLayer::add(::llvm::orc::JITDylib& dylib,
                              ::llvm::orc::ThreadSafeModule threadSafeModule,
                              ::llvm::orc::VModuleKey key) {
    auto name = dylib.getName();

    if (mLinkedModuleMap.count(name) == 0) {
        // fixme work correctly with contexts
        auto lock = threadSafeModule.getContextLock();
        auto ctx = std::make_unique<::llvm::LLVMContext>();
        ::llvm::orc::ThreadSafeContext tsc(std::move(ctx));
        auto llvmModule =
            std::make_unique<::llvm::Module>(name, *tsc.getContext());
        llvmModule->setSourceFileName("<null>");
        llvmModule->setDataLayout(
            threadSafeModule.getModule()->getDataLayout());
        llvmModule->setTargetTriple(
            threadSafeModule.getModule()->getTargetTriple());
        ::llvm::orc::ThreadSafeModule module(std::move(llvmModule),
                                             std::move(tsc));
        mLinkedModuleMap[name] = std::move(module);
        mMaterializedMap[name] = false;
    }
    {
        auto& myModule = mLinkedModuleMap[name];
        auto lock = threadSafeModule.getContextLock();
        auto ownedLock = myModule.getContextLock();
        auto clone = ::llvm::CloneModule(*threadSafeModule.getModule());
        ::llvm::Linker linker(*myModule.getModule());
        linker.linkInModule(std::move(clone));
    }

    return dylib.define(std::make_unique<MergeMaterializationUnit>(
        dylib.getExecutionSession(), *this, std::move(threadSafeModule), key));
}
MergeMaterializationUnit::MergeMaterializationUnit(
    ::llvm::orc::ExecutionSession& executionSession,
    MergeLayer& parent,
    ::llvm::orc::ThreadSafeModule module,
    ::llvm::orc::VModuleKey key)
    : ::llvm::orc::MaterializationUnit(::llvm::orc::SymbolFlagsMap(), key),
      mParent(parent) {
    ::llvm::orc::MangleAndInterner mangle(executionSession,
                                          module.getModule()->getDataLayout());

    auto lock = module.getContextLock();
    auto* plainModule = module.getModule();

    for (auto& globalValue : plainModule->global_values()) {
        if (globalValue.hasName() && !globalValue.isDeclaration() &&
            !globalValue.hasLocalLinkage() &&
            !globalValue.hasAvailableExternallyLinkage() &&
            !globalValue.hasAppendingLinkage()) {
            auto mangledName = mangle(globalValue.getName());
            SymbolFlags[mangledName] =
                ::llvm::JITSymbolFlags::fromGlobalValue(globalValue);
            //            SymbolToDefinition[MangledName] = &G;
        }
    }
}
void MergeMaterializationUnit::materialize(
    ::llvm::orc::MaterializationResponsibility R) {
    auto name = R.getTargetJITDylib().getName();

    if (!mParent.mMaterializedMap[name]) {
        auto& module = mParent.mLinkedModuleMap[name];
        mParent.emit(std::move(R), std::move(module));
    }
}
}  // namespace athena::backend::llvm