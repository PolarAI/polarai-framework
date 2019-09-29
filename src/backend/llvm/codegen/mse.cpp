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

#include "Mangler.h"
#include "common.h"

#include <athena/backend/llvm/LLVMGenerator.h>

namespace athena::backend::llvm::codegen {
template <typename T>
void registerMseImpl(LLVMGenerator *generator,
                     ::llvm::LLVMContext &ctx,
                     ::llvm::Module &module,
                     ::llvm::IRBuilder<> &builder,
                     core::inner::Tensor &a,
                     core::inner::Tensor &b,
                     core::inner::Tensor &c) {
    ::llvm::Function *calledFunction =
        generator->findLLVMFunction(Mangler::getMangledName<T>("mse"));

    if (!calledFunction) {
        core::FatalError(1, "Unknown function referenced");
    }

    std::vector<::llvm::Value *> ArgsV;

    ::llvm::Constant *device = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx),
        reinterpret_cast<size_t>(generator->getPreferredDevice("mse")));
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
}

void registerMse(LLVMGenerator *generator) {
    std::function<void(::llvm::LLVMContext &, ::llvm::Module &,
                       ::llvm::IRBuilder<> &, core::inner::Tensor &,
                       core::inner::Tensor &, core::inner::Tensor &)>
        f = [generator](::llvm::LLVMContext &ctx, ::llvm::Module &module,
                        ::llvm::IRBuilder<> &builder, core::inner::Tensor &a,
                        core::inner::Tensor &b, core::inner::Tensor &c) {
            if (a.getDataType() == core::DataType::FLOAT) {
                registerMseImpl<float>(generator, ctx, module, builder, a, b,
                                       c);
            } else if (a.getDataType() == core::DataType::DOUBLE) {
                registerMseImpl<double>(generator, ctx, module, builder, a, b,
                                        c);
            } else {
                new core::FatalError(1, "Unsupported type");
            }
        };

    generator->registerFunctor("mse", f);
}
}