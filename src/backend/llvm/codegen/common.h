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

#ifndef ATHENA_COMMON_H
#define ATHENA_COMMON_H

#include "../LLVMGenerator.h"

#include <functional>

namespace athena::backend::llvm::codegen {
void registerAllocate(LLVMGenerator* generator);
void registerFill(LLVMGenerator* generator);

template <typename T>
::llvm::Constant* getFPConstant(::llvm::LLVMContext& ctx, T value) {
  new utils::FatalError(utils::ATH_FATAL_OTHER, "Unable to create FP constant");
  return nullptr; // unreachable
}

template <>
::llvm::Constant* getFPConstant<float>(::llvm::LLVMContext& ctx, float value);
template <>
::llvm::Constant* getFPConstant<double>(::llvm::LLVMContext& ctx, double value);

using BuiltinThreeTensorArgs = std::function<void(
    ::llvm::LLVMContext&, ::llvm::Module&, ::llvm::IRBuilder<>&,
    core::internal::TensorInternal&, core::internal::TensorInternal&,
    core::internal::TensorInternal&)>;

using BuiltinThreeTensorWithOptsArgs = std::function<void(
    ::llvm::LLVMContext&, ::llvm::Module&, ::llvm::IRBuilder<>&, void*,
    core::internal::TensorInternal&, core::internal::TensorInternal&,
    core::internal::TensorInternal&)>;

using BuiltinTATATArgs = std::function<void(
    ::llvm::LLVMContext&, ::llvm::Module&, ::llvm::IRBuilder<>&,
    core::internal::TensorInternal&, uint64_t, core::internal::TensorInternal&,
    uint64_t, core::internal::TensorInternal&)>;

template <typename T>
void registerStandardBuiltin(const std::string& name, LLVMGenerator* generator);
} // namespace athena::backend::llvm::codegen

#endif // ATHENA_COMMON_H
