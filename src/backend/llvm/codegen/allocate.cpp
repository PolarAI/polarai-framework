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

void registerAllocate(LLVMGenerator* generator) {
  std::function<void(::llvm::LLVMContext&, ::llvm::Module&,
                     ::llvm::IRBuilder<>&, core::inner::Tensor&)>
      f = [generator](::llvm::LLVMContext& ctx, ::llvm::Module& module,
                      ::llvm::IRBuilder<>& builder, core::inner::Tensor& a) {
        ::llvm::Function* calledFunction =
            generator->findLLVMFunction("athn_allocate_v");

        if (!calledFunction) {
          core::FatalError(core::ATH_FATAL_OTHER,
                           "Unknown function referenced");
        }

        std::vector<::llvm::Value*> ArgsV;
        ::llvm::Constant* device = ::llvm::ConstantInt::get(
            ::llvm::Type::getInt64Ty(ctx),
            reinterpret_cast<size_t>(
                generator->getPreferredDevice("allocate")));
        ArgsV.push_back(device);
        ::llvm::Constant* allocatorConst =
            ::llvm::ConstantInt::get(::llvm::Type::getInt64Ty(ctx),
                                     (size_t)(&generator->getAllocator()));
        ArgsV.push_back(allocatorConst);
        ::llvm::Constant* tensorConst = ::llvm::ConstantInt::get(
            ::llvm::Type::getInt64Ty(ctx), (size_t)(&a));
        ArgsV.push_back(tensorConst);
        auto callInst = builder.CreateCall(calledFunction, ArgsV);
        if (!callInst) {
          new core::FatalError(core::ATH_FATAL_OTHER,
                               "Call instruction for allocate is not created");
        }
      };

  generator->registerFunctor("allocate", f);
}
} // namespace athena::backend::llvm::codegen