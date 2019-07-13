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

#ifndef ATHENA_ATHENAJIT_H
#define ATHENA_ATHENAJIT_H

#include <llvm/ADT/StringRef.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/IRTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <memory>

namespace athena::backend::llvm {
/**
 * Execute LLVM IR
 */
class AthenaJIT {
    private:
    ::llvm::orc::ExecutionSession mExecutionSession;
    ::llvm::orc::RTDyldObjectLinkingLayer mObjectLayer;
    ::llvm::orc::IRCompileLayer mCompileLayer;
    ::llvm::orc::IRTransformLayer mOptimizeLayer;

    ::llvm::DataLayout mDataLayout;
    ::llvm::orc::MangleAndInterner mMangle;
    ::llvm::orc::ThreadSafeContext mContext;

    static ::llvm::Expected<::llvm::orc::ThreadSafeModule> optimizeModule(
        ::llvm::orc::ThreadSafeModule TSM,
        const ::llvm::orc::MaterializationResponsibility &R);

    public:
    AthenaJIT(::llvm::orc::JITTargetMachineBuilder JTMB,
              ::llvm::DataLayout &&DL);

    static std::unique_ptr<AthenaJIT> create();
    const ::llvm::DataLayout &getDataLayout() const {
        return mDataLayout;
    }
    ::llvm::LLVMContext &getContext() {
        return *mContext.getContext();
    }

    ::llvm::Error addModule(std::unique_ptr<::llvm::Module> &M);
    ::llvm::Expected<::llvm::JITEvaluatedSymbol> lookup(::llvm::StringRef Name);
};
}  // namespace athena::backend::llvm

#endif  // ATHENA_ATHENAJIT_H
