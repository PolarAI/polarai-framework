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

#include "LLVMGenerator.h"

#include "codegen/register_default_functors.h"
#include "utils.h"

#include <athena/core/FatalError.h>
#include <athena/loaders/MemoryLoader/MemoryLoader.h>

#include <utility>

namespace athena::backend::llvm {

llvm::LLVMGenerator::LLVMGenerator(
    ::llvm::LLVMContext& ctx, const std::unique_ptr<::llvm::Module>& module,
    core::Allocator& allocator,
    std::vector<std::unique_ptr<::llvm::Module>>& existing,
    std::unordered_map<std::string_view, Device*> map)
    : mGeneratedModule(module), mCurrentMainBlock(nullptr),
      mCurrentBlock(mCurrentMainBlock), mContext(ctx),
      mBuilder(::llvm::IRBuilder(ctx)), mAllocator(allocator),
      mExistingModules(existing), mCurrentPreferredDevice(nullptr),
      mGraphMap(std::move(map)) {
  mBuilder.SetInsertPoint(mCurrentMainBlock);
  codegen::registerDefaultFunctors(this);
}

::llvm::IRBuilder<>& LLVMGenerator::getBuilder() { return mBuilder; }

void LLVMGenerator::generateLoad(const core::AbstractLoader& loader,
                                 core::inner::Tensor& tensor) {
  ::llvm::Function* loadFunction =
      mGeneratedModule->getFunction(loader.getLoadCName());

  if (!loadFunction) {
    std::vector<::llvm::Type*> args(3, ::llvm::Type::getInt64Ty(mContext));
    ::llvm::FunctionType* FT = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(mContext), args, false);

    loadFunction =
        ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage,
                                 loader.getLoadCName(), mGeneratedModule.get());
  }

  if (!loadFunction) {
    core::FatalError(core::ATH_FATAL_OTHER, "Unknown function referenced");
  }

  std::vector<::llvm::Value*> ArgsV;
  ::llvm::Constant* loaderConst = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(mContext), reinterpret_cast<size_t>(&loader));
  ArgsV.push_back(loaderConst);
  ::llvm::Constant* allocatorConst =
      ::llvm::ConstantInt::get(::llvm::Type::getInt64Ty(mContext),
                               reinterpret_cast<size_t>(&mAllocator));
  ArgsV.push_back(allocatorConst);
  ::llvm::Constant* tensorConst = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(mContext), reinterpret_cast<size_t>(&tensor));
  ArgsV.push_back(tensorConst);
  mBuilder.CreateCall(loadFunction, ArgsV);
}
void LLVMGenerator::generateImpl(std::string& name, core::inner::Tensor& a) {
  mFunctorsMap[name](mContext, *mGeneratedModule, mBuilder, a);
}
void LLVMGenerator::generateImpl(std::string& name, core::inner::Tensor& a,
                                 void*& b) {
  mFunctorsMap[name](mContext, *mGeneratedModule, mBuilder, a, b);
}
void LLVMGenerator::generateImpl(std::string& name, core::inner::Tensor& a,
                                 core::inner::Tensor& b) {
  mFunctorsMap[name](mContext, *mGeneratedModule, mBuilder, a, b);
}
void LLVMGenerator::generateImpl(std::string& name, core::inner::Tensor& a,
                                 core::inner::Tensor& b,
                                 core::inner::Tensor& c) {
  mFunctorsMap[name](mContext, *mGeneratedModule, mBuilder, a, b, c);
}
void LLVMGenerator::generateImpl(std::string& name, core::inner::Tensor& a,
                                 uint64_t scaleA, core::inner::Tensor& b,
                                 uint64_t scaleB, core::inner::Tensor& c) {
  mFunctorsMap[name](mContext, *mGeneratedModule, mBuilder, a, scaleA, b,
                     scaleB, c);
}
void LLVMGenerator::generateImpl(std::string& name, void* options,
                                 core::inner::Tensor& a, core::inner::Tensor& b,
                                 core::inner::Tensor& c) {
  mFunctorsMap[name](mContext, *mGeneratedModule, mBuilder, options, a, b, c);
}
void LLVMGenerator::openNode(std::string_view name) {
#ifdef DEBUG
  assert(mCurrentBlock == mCurrentMainBlock && "There is an opened node");
#endif
  mCurrentPreferredDevice = mGraphMap[name];
  ::llvm::FunctionType* FT =
      ::llvm::FunctionType::get(::llvm::Type::getVoidTy(mContext), false);
  auto nodeFunction =
      ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage,
                               "node_" + std::string(name), *mGeneratedModule);

  mBuilder.CreateCall(nodeFunction);

  mCurrentBlock = ::llvm::BasicBlock::Create(mContext, "entry", nodeFunction);

  mBuilder.SetInsertPoint(mCurrentBlock);
}
void LLVMGenerator::closeNode() {
  mBuilder.CreateRetVoid();
  mCurrentBlock = mCurrentMainBlock;
  mBuilder.SetInsertPoint(mCurrentBlock);
}
void LLVMGenerator::generateFunctionHeader(const std::string& name) {
  ::llvm::FunctionType* FT =
      ::llvm::FunctionType::get(::llvm::Type::getVoidTy(mContext), false);
  ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage, name,
                           *mGeneratedModule);
  mCurrentMainBlock = ::llvm::BasicBlock::Create(
      mContext, "entry", mGeneratedModule->getFunction(name));
  mCurrentBlock = mCurrentMainBlock;
  mBuilder.SetInsertPoint(mCurrentBlock);
}
void LLVMGenerator::generateFunctionFooter() {
  mBuilder.CreateRetVoid();
  mCurrentBlock = nullptr;
  mCurrentMainBlock = nullptr;
}
} // namespace athena::backend::llvm