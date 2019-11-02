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

#ifndef ATHENA_LLVMGENERATORFUNCTOR_H
#define ATHENA_LLVMGENERATORFUNCTOR_H

#include <athena/backend/llvm/llvm_export.h>
#include <athena/core/FatalError.h>
#include <athena/core/inner/Tensor.h>

#include <any>
#include <functional>

namespace athena::backend::llvm {
template <typename Ret>
struct ATH_BACKEND_LLVM_EXPORT LLVMGeneratorFunctor {
    LLVMGeneratorFunctor() = default;
    template <typename F>
    LLVMGeneratorFunctor(F&& fun) : LLVMGeneratorFunctor(std::function(fun)){};
    template <typename... Args>
    LLVMGeneratorFunctor(std::function<Ret(Args...)> fun)
        : mFunctor(std::move(fun)) {}
    template <typename... Args>
    Ret operator()(Args&&... args) {
        return std::invoke(
            std::any_cast<std::function<Ret(Args && ...)>>(mFunctor),
            std::forward<Args&&>(args)...);
    }

    private:
    std::any mFunctor;
};

template <>
struct ATH_BACKEND_LLVM_EXPORT LLVMGeneratorFunctor<void> {
    LLVMGeneratorFunctor() = default;
    template <typename F>
    explicit LLVMGeneratorFunctor(F&& fun)
        : LLVMGeneratorFunctor(std::function(fun)){};
    template <typename... Args>
    explicit LLVMGeneratorFunctor(std::function<void(Args...)> fun)
        : mFunctor(std::move(fun)) {}
    template <typename... Args>
    void operator()(Args&&... args) {
        return std::invoke(
            std::any_cast<std::function<void(Args && ...)>>(mFunctor),
            std::forward<Args&&>(args)...);
    }

    private:
    std::any mFunctor;
};
}  // namespace athena::backend::llvm

#endif  // ATHENA_LLVMGENERATORFUNCTOR_H
