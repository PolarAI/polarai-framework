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

#include <athena/core/FatalError.h>

#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace athena::backend::llvm {

void MergeLayer::emit(::llvm::orc::MaterializationResponsibility responsibility,
                      ::llvm::orc::ThreadSafeModule threadSafeModule) {
  athena_assert((bool)threadSafeModule, "Module must not be null");
  mMaterializedMap[responsibility.getTargetJITDylib().getName()] = true;
  mBaseLayer.emit(std::move(responsibility), std::move(threadSafeModule));
}

::llvm::Error MergeLayer::add(::llvm::orc::JITDylib& dylib,
                              ::llvm::orc::ThreadSafeModule threadSafeModule,
                              ::llvm::orc::VModuleKey key) {
  auto name = dylib.getName();

  if (mLinkedModuleMap.count(name) == 0) {
    // fixme work correctly with contexts
    threadSafeModule.withModuleDo([&](::llvm::Module& module) {
      auto ctx = std::make_unique<::llvm::LLVMContext>();
      ::llvm::orc::ThreadSafeContext tsc(std::move(ctx));
      auto llvmModule =
          std::make_unique<::llvm::Module>(name, *tsc.getContext());
      llvmModule->setSourceFileName("<null>");
      llvmModule->setDataLayout(module.getDataLayout());
      llvmModule->setTargetTriple(module.getTargetTriple());
      ::llvm::orc::ThreadSafeModule tsModule(std::move(llvmModule),
                                             std::move(tsc));
      mLinkedModuleMap[name] = std::move(tsModule);
      mMaterializedMap[name] = false;
    });
    //        auto lock = threadSafeModule.getContextLock();
  }
  auto& myModule = mLinkedModuleMap[name];
  myModule.withModuleDo([&](::llvm::Module& module1) {
    threadSafeModule.withModuleDo([&](::llvm::Module& module2) {
      auto clone = ::llvm::CloneModule(module2);
      ::llvm::Linker linker(module1);
      linker.linkInModule(std::move(clone));
    });
  });

  return dylib.define(std::make_unique<MergeMaterializationUnit>(
      dylib.getExecutionSession(), *this, std::move(threadSafeModule), key));
}
MergeMaterializationUnit::MergeMaterializationUnit(
    ::llvm::orc::ExecutionSession& executionSession, MergeLayer& parent,
    ::llvm::orc::ThreadSafeModule module, ::llvm::orc::VModuleKey key)
    : ::llvm::orc::MaterializationUnit(::llvm::orc::SymbolFlagsMap(), key),
      mParent(parent) {
  module.withModuleDo([&](::llvm::Module& plainModule) {
    ::llvm::orc::MangleAndInterner mangle(executionSession,
                                          plainModule.getDataLayout());
    for (auto& globalValue : plainModule.global_values()) {
      if (globalValue.hasName() && !globalValue.isDeclaration() &&
          !globalValue.hasLocalLinkage() &&
          !globalValue.hasAvailableExternallyLinkage() &&
          !globalValue.hasAppendingLinkage()) {
        auto mangledName = mangle(globalValue.getName());
        SymbolFlags[mangledName] =
            ::llvm::JITSymbolFlags::fromGlobalValue(globalValue);
      }
    }
  });
}
void MergeMaterializationUnit::materialize(
    ::llvm::orc::MaterializationResponsibility R) {
  auto name = R.getTargetJITDylib().getName();

  if (!mParent.mMaterializedMap[name]) {
    auto& module = mParent.mLinkedModuleMap[name];
    mParent.emit(std::move(R), std::move(module));
  }
}
} // namespace athena::backend::llvm