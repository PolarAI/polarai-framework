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

#include "common.h"

namespace athena::backend::llvm::codegen {

void registerFill(LLVMGenerator* generator) {
  std::function<void(::llvm::LLVMContext&, ::llvm::Module&,
                     ::llvm::IRBuilder<>&, core::internal::TensorInternal&,
                     void*&)>
      f = [generator](::llvm::LLVMContext& ctx, ::llvm::Module& module,
                      ::llvm::IRBuilder<>& builder,
                      core::internal::TensorInternal& a, void*& filler) {
        // todo handle different data types

        ::llvm::Function* calledFunction =
            generator->findLLVMFunction("athn_fill_f");

        if (!calledFunction) {
          utils::FatalError(utils::ATH_FATAL_OTHER,
                            "Unknown function referenced");
        }

        std::vector<::llvm::Value*> ArgsV;
        ::llvm::Constant* device = ::llvm::ConstantInt::get(
            ::llvm::Type::getInt64Ty(ctx),
            reinterpret_cast<size_t>(generator->getPreferredDevice("fill")));
        ArgsV.push_back(device);
        ::llvm::Constant* allocatorConst =
            ::llvm::ConstantInt::get(::llvm::Type::getInt64Ty(ctx),
                                     (size_t)(&generator->getAllocator()));
        ArgsV.push_back(allocatorConst);
        ::llvm::Constant* tensorConst = ::llvm::ConstantInt::get(
            ::llvm::Type::getInt64Ty(ctx), (size_t)(&a));
        ArgsV.push_back(tensorConst);
        ::llvm::Constant* fillerConst = ::llvm::ConstantFP::get(
            ::llvm::Type::getFloatTy(ctx), *reinterpret_cast<float*>(filler));
        ArgsV.push_back(fillerConst);
        builder.CreateCall(calledFunction, ArgsV);
      };
  generator->registerFunctor("fill", f);
}
} // namespace athena::backend::llvm::codegen
