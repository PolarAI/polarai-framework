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

#ifndef ATHENA_MERGELAYER_H
#define ATHENA_MERGELAYER_H

#include <athena/backend/llvm/llvm_export.h>

#include <llvm/ExecutionEngine/Orc/Layer.h>
#include <llvm/ExecutionEngine/Orc/LazyReexports.h>

namespace athena::backend::llvm {

class MergeMaterializationUnit;

class ATH_BACKEND_LLVM_EXPORT MergeLayer : public ::llvm::orc::IRLayer {
  friend MergeMaterializationUnit;

private:
  ::llvm::orc::IRLayer& mBaseLayer;
  std::map<std::string, ::llvm::orc::ThreadSafeModule> mLinkedModuleMap;
  std::map<std::string, bool> mMaterializedMap;

public:
  MergeLayer(::llvm::orc::ExecutionSession& ES, ::llvm::orc::IRLayer& baseLayer)
      : IRLayer(ES), mBaseLayer(baseLayer){};
  ~MergeLayer() override = default;
  ::llvm::Error add(::llvm::orc::JITDylib& dylib,
                    ::llvm::orc::ThreadSafeModule threadSafeModule,
                    ::llvm::orc::VModuleKey key) override;
  void emit(::llvm::orc::MaterializationResponsibility responsibility,
            ::llvm::orc::ThreadSafeModule threadSafeModule) override;
};

class ATH_BACKEND_LLVM_EXPORT MergeMaterializationUnit
    : public ::llvm::orc::MaterializationUnit {
protected:
  MergeLayer& mParent;

public:
  MergeMaterializationUnit(::llvm::orc::ExecutionSession& executionSession,
                           MergeLayer& parent,
                           ::llvm::orc::ThreadSafeModule module,
                           ::llvm::orc::VModuleKey key);

  [[nodiscard]] ::llvm::StringRef getName() const override {
    // todo return module name
    return "MergeMaterializationUnit";
  }

private:
  void materialize(::llvm::orc::MaterializationResponsibility R) override;
  void discard(const ::llvm::orc::JITDylib& JD,
               const ::llvm::orc::SymbolStringPtr& Name) override {
    // todo there's no way to remove symbols
  }
};
} // namespace athena::backend::llvm

#endif // ATHENA_MERGELAYER_H
