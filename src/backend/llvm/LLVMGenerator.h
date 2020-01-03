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
#ifndef ATHENA_LLVMGENERATOR_H
#define ATHENA_LLVMGENERATOR_H

#include <athena/backend/llvm/LLVMGeneratorFunctor.h>
#include <athena/backend/llvm/runtime/Device.h>
#include <athena/core/AbstractGenerator.h>
#include <athena/core/AbstractLoader.h>
#include <athena/core/Allocator.h>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <map>
#include <unordered_map>

namespace athena::backend::llvm {
/**
 * LLVM-based code generator
 */
class LLVMGenerator : public core::AbstractGenerator {
private:
  std::map<std::string, LLVMGeneratorFunctor<void>> mFunctorsMap;
  const std::unique_ptr<::llvm::Module>& mGeneratedModule;
  std::vector<std::unique_ptr<::llvm::Module>>& mExistingModules;
  ::llvm::LLVMContext& mContext;
  // todo abatashev: refactor main block
  ::llvm::BasicBlock* mCurrentMainBlock;
  ::llvm::BasicBlock* mCurrentBlock;
  ::llvm::IRBuilder<> mBuilder;

  Device* mCurrentPreferredDevice;

  core::Allocator& mAllocator;

  std::unordered_map<std::string_view, Device*> mGraphMap;

protected:
  void generateImpl(std::string&, core::inner::Tensor& a) override;
  void generateImpl(std::string&, core::inner::Tensor& a, void*& b) override;
  void generateImpl(std::string&, core::inner::Tensor& a,
                    core::inner::Tensor& b) override;
  void generateImpl(std::string&, core::inner::Tensor& a,
                    core::inner::Tensor& b, core::inner::Tensor& c) override;
  void generateImpl(std::string&, core::inner::Tensor& a, uint64_t scaleA,
                    core::inner::Tensor& b, uint64_t scaleB,
                    core::inner::Tensor& c) override;
  void generateImpl(std::string&, void*, core::inner::Tensor& a,
                    core::inner::Tensor& b, core::inner::Tensor& c) override;

public:
  explicit LLVMGenerator(::llvm::LLVMContext& ctx,
                         const std::unique_ptr<::llvm::Module>& module,
                         core::Allocator& allocator,
                         std::vector<std::unique_ptr<::llvm::Module>>& existing,
                         std::unordered_map<std::string_view, Device*> map);
  /**
   * Generate code to execute loaders subroutines
   * @param loader Loader to be used
   * @param tensor Destination Tensor
   */
  void generateLoad(const core::AbstractLoader& loader,
                    core::inner::Tensor& tensor) override;
  /**
   *
   * @return LLVM IR Builder
   */
  ::llvm::IRBuilder<>& getBuilder();

  /**
   * Notifies generator that Node code generation begins
   * @param name Node name
   */
  void openNode(std::string_view name) override;
  /**
   * Notifies generator that Node code generation ends
   */
  void closeNode() override;

  /**
   * Creates empty function without arguments and sets it as current main
   * block
   * @param name Function name
   */
  void generateFunctionHeader(const std::string& name) override;

  /**
   * Generates return command for current function and removes it from current
   * main block
   */
  void generateFunctionFooter() override;

  /**
   * Register new functor
   * @tparam Args Functor arguments
   * @param name Functor name
   * @param f Functor function
   */
  template <typename... Args>
  void registerFunctor(const std::string& name,
                       std::function<void(Args...)>& f) {
    if (mFunctorsMap.count(name)) {
      core::FatalError(core::ATH_FATAL_OTHER,
                       "Functor already registered: " + name);
    }
    mFunctorsMap[name] = LLVMGeneratorFunctor(f);
  }

  /**
   * Remove Functor from registry
   * @param name Functor name
   */
  void unregisterFunctor(std::string& name) {
    if (mFunctorsMap.count(name)) {
      mFunctorsMap.erase(mFunctorsMap.find(name));
    }
  }

  /**
   *
   * @return Associated Allocator
   */
  core::Allocator& getAllocator() { return mAllocator; }

  void
  setExistingModules(std::vector<std::unique_ptr<::llvm::Module>>&& modules) {
    mExistingModules = std::move(modules);
  }

  Device* getPreferredDevice(const std::string&) {
    return mCurrentPreferredDevice;
  }

  ::llvm::Function* findLLVMFunction(const std::string& name) {
    for (const auto& module : mExistingModules) {
      auto func = module->getFunction(name);
      if (func)
        return func;
    }
    return nullptr;
  }
};
} // namespace athena::backend::llvm

#endif // ATHENA_LLVMGENERATOR_H
