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

#ifndef ATHENA_ATHENAJIT_H
#define ATHENA_ATHENAJIT_H

#include <athena/backend/llvm/llvm_export.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h>
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
#include <llvm/IR/PassManager.h>
#include <llvm/Support/ThreadPool.h>

#include <atomic>
#include <memory>
#include <thread>

namespace athena::backend::llvm {

/// AthenaJIT executes generated graph through LLVM's ORC.
///
/// The following layers are used during compilation phase:
/// 1. Compile On Demand layer with whole module partitioning
/// 2. IR transformation layer with O2 optimization pipeline
/// 3. IR compilation layer
/// 4. Object layer
///
/// Although graph consists of a single translation unit, the compilation
/// is done in multiple threads to assist the case when we add runtime modules.
class ATH_BACKEND_LLVM_EXPORT AthenaJIT {
private:
  ::llvm::orc::ExecutionSession mExecutionSession;

  std::unique_ptr<::llvm::orc::LazyCallThroughManager> mCallThroughManager;

  std::unique_ptr<::llvm::orc::RTDyldObjectLinkingLayer> mObjectLayer;
  std::unique_ptr<::llvm::orc::IRCompileLayer> mCompileLayer;
  std::unique_ptr<::llvm::orc::IRTransformLayer> mOptimizeLayer;
  std::unique_ptr<::llvm::orc::CompileOnDemandLayer> mCODLayer;

  ::llvm::DataLayout mDataLayout;

  ::llvm::orc::MangleAndInterner mMangle;
  ::llvm::orc::JITDylib& mMainJD;
  ::llvm::orc::ThreadSafeContext mContext;

  ::llvm::ThreadPool mCompileThreads{::llvm::hardware_concurrency()};

  ::llvm::orc::ImplSymbolMap mSymbolMap;

  static ::llvm::Expected<::llvm::orc::ThreadSafeModule>
  optimizeModule(::llvm::orc::ThreadSafeModule TSM,
                 const ::llvm::orc::MaterializationResponsibility&
                     materializationResponsibility);

  void setUpJITDylib(::llvm::orc::JITDylib* jitDylib);

public:
  AthenaJIT(::llvm::orc::JITTargetMachineBuilder JTMB,
            ::llvm::DataLayout&& materializationUnit);
  AthenaJIT(const AthenaJIT&) = delete;
  AthenaJIT& operator=(const AthenaJIT&) = delete;
  ~AthenaJIT();

  /// \return an instance of default AthenaJIT.
  static std::unique_ptr<AthenaJIT> create();

  const ::llvm::DataLayout& getDataLayout() const { return mDataLayout; }
  ::llvm::LLVMContext& getContext() { return *mContext.getContext(); }

  /// Adds module to the compilation queue.
  ///
  /// The real compilation takes place only when module is requested.
  ///
  /// \param module is a unique pointer to the module being added.
  /// \return non-empty error if addition failed.
  ::llvm::Error addModule(std::unique_ptr<::llvm::Module>& module);

  /// Finds a named symbol in compiled code.
  ///
  /// \param name is a name of JITTed symbol to find.
  /// \return a pointer to compiled symbol.
  ::llvm::Expected<::llvm::JITEvaluatedSymbol> lookup(::llvm::StringRef name);
};
} // namespace athena::backend::llvm

#endif // ATHENA_ATHENAJIT_H
