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

#include "Mangler.h"
#include "common.h"

#include <athena/backend/llvm/LLVMGenerator.h>
namespace athena::backend::llvm::codegen {

void registerAdd(LLVMGenerator *generator) {
    std::function<void(::llvm::LLVMContext &, ::llvm::Module &,
                       ::llvm::IRBuilder<> &, core::inner::Tensor &,
                       core::inner::Tensor &, core::inner::Tensor &)>
        f = [generator](::llvm::LLVMContext &ctx, ::llvm::Module &module,
                        ::llvm::IRBuilder<> &builder, core::inner::Tensor &a,
                        core::inner::Tensor &b, core::inner::Tensor &c) {
            // todo handle different data types

            ::llvm::Function *calledFunction =
                generator->findLLVMFunction("athn_add_f");

            if (!calledFunction) {
                core::FatalError(1, "Unknown function referenced");
            }

            // todo check arg count

            std::vector<::llvm::Value *> ArgsV;

            ::llvm::Constant *device = ::llvm::ConstantInt::get(
                ::llvm::Type::getInt64Ty(ctx),
                reinterpret_cast<size_t>(generator->getPreferredDevice("add")));
            ArgsV.push_back(device);
            ::llvm::Constant *allocator = ::llvm::ConstantInt::get(
                ::llvm::Type::getInt64Ty(ctx),
                reinterpret_cast<size_t>(&generator->getAllocator()));
            ArgsV.push_back(allocator);
            ::llvm::Constant *aTensor = ::llvm::ConstantInt::get(
                ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&a));
            ArgsV.push_back(aTensor);
            ::llvm::Constant *bTensor = ::llvm::ConstantInt::get(
                ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&b));
            ArgsV.push_back(bTensor);
            ::llvm::Constant *cTensor = ::llvm::ConstantInt::get(
                ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&c));
            ArgsV.push_back(cTensor);
            builder.CreateCall(calledFunction, ArgsV);
        };

    generator->registerFunctor("add", f);
}

template <typename T>
void registerFmaImpl(LLVMGenerator *generator,
                     ::llvm::LLVMContext &ctx,
                     ::llvm::Module &module,
                     ::llvm::IRBuilder<> &builder,
                     core::inner::Tensor &a,
                     uint64_t scaleA,
                     core::inner::Tensor &b,
                     uint64_t scaleB,
                     core::inner::Tensor &c) {
    auto realScaleA = *reinterpret_cast<T *>(&scaleA);
    auto realScaleB = *reinterpret_cast<T *>(&scaleB);

    ::llvm::Function *calledFunction =
        generator->findLLVMFunction(Mangler::getMangledName<T>("fma"));

    if (!calledFunction) {
        core::FatalError(1, "Unknown function referenced");
    }

    std::vector<::llvm::Value *> ArgsV;

    ::llvm::Constant *device = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx),
        reinterpret_cast<size_t>(generator->getPreferredDevice("fma")));
    ArgsV.push_back(device);
    ::llvm::Constant *allocator = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx),
        reinterpret_cast<size_t>(&generator->getAllocator()));
    ArgsV.push_back(allocator);
    ::llvm::Constant *aTensor = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&a));
    ArgsV.push_back(aTensor);
    ::llvm::Constant *scaleAConst = getFPConstant<T>(ctx, realScaleA);
    ArgsV.push_back(scaleAConst);
    ::llvm::Constant *bTensor = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&b));
    ArgsV.push_back(bTensor);
    ::llvm::Constant *scaleBConst = getFPConstant<T>(ctx, realScaleB);
    ArgsV.push_back(scaleBConst);
    ::llvm::Constant *cTensor = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&c));
    ArgsV.push_back(cTensor);
    builder.CreateCall(calledFunction, ArgsV);
}

void registerFma(LLVMGenerator *generator) {
    std::function<void(::llvm::LLVMContext &, ::llvm::Module &,
                       ::llvm::IRBuilder<> &, core::inner::Tensor &, uint64_t &,
                       core::inner::Tensor &, uint64_t &,
                       core::inner::Tensor &)>
        f = [generator](::llvm::LLVMContext &ctx, ::llvm::Module &module,
                        ::llvm::IRBuilder<> &builder, core::inner::Tensor &a,
                        uint64_t scaleA, core::inner::Tensor &b,
                        uint64_t scaleB, core::inner::Tensor &c) {
            if (a.getDataType() == core::DataType::FLOAT) {
                registerFmaImpl<float>(generator, ctx, module, builder, a,
                                       scaleA, b, scaleB, c);
            } else if (a.getDataType() == core::DataType::DOUBLE) {
                registerFmaImpl<double>(generator, ctx, module, builder, a,
                                        scaleA, b, scaleB, c);
            } else {
                new core::FatalError(1, "Unsupported type");
            }
        };

    generator->registerFunctor("fma", f);
}
}  // namespace athena::backend::llvm::codegen
