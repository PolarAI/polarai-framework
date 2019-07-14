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

#include "codegen/register_default_functors.h"
#include "utils.h"

#include <athena/backend/llvm/LLVMGenerator.h>
#include <athena/core/FatalError.h>
#include <athena/loaders/MemoryLoader/MemoryLoader.h>

namespace athena::backend::llvm {

llvm::LLVMGenerator::LLVMGenerator(
    ::llvm::LLVMContext &ctx,
    const std::unique_ptr<::llvm::Module> &module,
    core::Allocator &allocator,
    std::vector<std::unique_ptr<::llvm::Module>> &existing)
    : mGeneratedModule(module),
      mMainBlock(::llvm::BasicBlock::Create(
          ctx, "entry", module->getFunction("jitmain"))),
      mCurrentBlock(mMainBlock),
      mContext(ctx),
      mBuilder(::llvm::IRBuilder(mMainBlock)),
      mAllocator(allocator),
      mExistingModules(existing) {
    mBuilder.SetInsertPoint(mMainBlock);
    codegen::registerDefaultFunctors(this);
}

::llvm::IRBuilder<> &LLVMGenerator::getBuilder() {
    return mBuilder;
}

void LLVMGenerator::generateLoad(const core::AbstractLoader &loader,
                                 core::inner::Tensor &tensor) {
    ::llvm::Function *loadFunction =
        mGeneratedModule->getFunction(loader.getLoadCName());

    if (!loadFunction) {
        std::vector<::llvm::Type *> args(3, ::llvm::Type::getInt64Ty(mContext));
        ::llvm::FunctionType *FT = ::llvm::FunctionType::get(
            ::llvm::Type::getVoidTy(mContext), args, false);

        loadFunction = ::llvm::Function::Create(
            FT, ::llvm::Function::ExternalLinkage, loader.getLoadCName(),
            mGeneratedModule.get());
    }

    if (!loadFunction) {
        core::FatalError(1, "Unknown function referenced");
    }

    std::vector<::llvm::Value *> ArgsV;
    ::llvm::Constant *loaderConst = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(mContext), reinterpret_cast<size_t>(&loader));
    ArgsV.push_back(loaderConst);
    ::llvm::Constant *allocatorConst =
        ::llvm::ConstantInt::get(::llvm::Type::getInt64Ty(mContext),
                                 reinterpret_cast<size_t>(&mAllocator));
    ArgsV.push_back(allocatorConst);
    ::llvm::Constant *tensorConst = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(mContext), reinterpret_cast<size_t>(&tensor));
    ArgsV.push_back(tensorConst);
    mBuilder.CreateCall(loadFunction, ArgsV);
}
void LLVMGenerator::generateImpl(std::string &name, core::inner::Tensor &a) {
    mFunctorsMap[name](mContext, *mGeneratedModule, mBuilder, a);
}
void LLVMGenerator::generateImpl(std::string &name,
                                 core::inner::Tensor &a,
                                 void *&b) {
    mFunctorsMap[name](mContext, *mGeneratedModule, mBuilder, a, b);
}
void LLVMGenerator::generateImpl(std::string &name,
                                 core::inner::Tensor &a,
                                 core::inner::Tensor &b) {
    mFunctorsMap[name](mContext, *mGeneratedModule, mBuilder, a, b);
}
void LLVMGenerator::generateImpl(std::string &name,
                                 core::inner::Tensor &a,
                                 core::inner::Tensor &b,
                                 core::inner::Tensor &c) {
    mFunctorsMap[name](mContext, *mGeneratedModule, mBuilder, a, b, c);
}
void LLVMGenerator::openNode(std::string_view name) {
#ifdef DEBUG
    assert(mCurrentBlock == mMainBlock && "There is an opened node");
#endif
    ::llvm::FunctionType *FT =
        ::llvm::FunctionType::get(::llvm::Type::getVoidTy(mContext), false);
    auto nodeFunction = ::llvm::Function::Create(
        FT, ::llvm::Function::ExternalLinkage, "node_" + std::string(name),
        *mGeneratedModule);

    mBuilder.CreateCall(nodeFunction);

    mCurrentBlock = ::llvm::BasicBlock::Create(mContext, "entry", nodeFunction);

    mBuilder.SetInsertPoint(mCurrentBlock);
}
void LLVMGenerator::closeNode() {
    mBuilder.CreateRetVoid();
    mCurrentBlock = mMainBlock;
    mBuilder.SetInsertPoint(mCurrentBlock);
}
}  // namespace athena::backend::llvm