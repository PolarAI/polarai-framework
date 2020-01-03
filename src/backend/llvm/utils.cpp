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

#include "utils.h"

namespace athena::backend::llvm::impl {

::llvm::Function* create_get_fast_pointer_decl(::llvm::LLVMContext& ctx,
                                               ::llvm::Module& module) {
  std::vector<::llvm::Type*> args(2, ::llvm::Type::getInt64Ty(ctx));
  ::llvm::FunctionType* FT =
      ::llvm::FunctionType::get(::llvm::Type::getInt64Ty(ctx), args, false);

  ::llvm::Function* F =
      ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage,
                               "athena_get_fast_pointer", &module);

  return F;
}

} // namespace athena::backend::llvm::impl