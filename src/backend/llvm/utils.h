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

#ifndef ATHENA_UTILS_H
#define ATHENA_UTILS_H

#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

namespace athena::backend::llvm::impl {

::llvm::Function* create_get_fast_pointer_decl(::llvm::LLVMContext& ctx,
                                               ::llvm::Module& module);

} // namespace athena::backend::llvm::impl

#endif // ATHENA_UTILS_H
