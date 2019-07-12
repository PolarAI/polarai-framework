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

#ifndef ATHENA_RUNTIME_DRIVER_H
#define ATHENA_RUNTIME_DRIVER_H

#include <athena/core/FatalError.h>

#include <dlfcn.h>
#include <string>
#include <string_view>

#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"


namespace athena::backend::llvm {
class RuntimeDriver {
private:
    void *mLibraryHandle;
    void (*mFaddPointer)(void *a, size_t ca, void *b, size_t cb, void *c);
    void (*mAllocatePointer)(void *a, void *t);
    void *(*mGetFPPointer)(void *a, void *t);
    void (*mFfillPointer)(void *allocator, void *tensor, float f);

    void *getFunction(std::string_view nameFunction);

    std::vector<std::unique_ptr<::llvm::Module>> mModules;

    void generateLLLVMIrBindings(
        ::llvm::LLVMContext &ctx,
        ::llvm::Module &module,
        ::llvm::IRBuilder<> &builder
    );

    void *getFunctionPtr(std::string str) { return nullptr; }

    ::llvm::ArrayRef<::llvm::Value*> getArgs(::llvm::Function *function);

public:
    RuntimeDriver();
    explicit RuntimeDriver(std::string_view nameLibrary);
    RuntimeDriver(const RuntimeDriver &rhs) = delete;
    RuntimeDriver(RuntimeDriver &&rhs) noexcept = default;
    ~RuntimeDriver();

    RuntimeDriver &operator=(const RuntimeDriver &rhs) = delete;
    RuntimeDriver &operator=(RuntimeDriver &&rhs) noexcept;

    void load(std::string_view nameLibrary);
    void unload();
    void reload(std::string_view nameLibrary);
    bool isLoaded() const;

    void athena_fadd(void *a, size_t ca, void *b, size_t cb, void *c) {
        mFaddPointer(a, ca, b, cb, c);
    }
    void athena_allocate(void *a, void *t) {
        mAllocatePointer(a, t);
    }
    void *athena_get_fast_pointer(void *a, void *t) {
        return mGetFPPointer(a, t);
    }
    void athena_ffill(void *allocator, void *tensor, float f) {}
};
extern RuntimeDriver kRuntimeDriver;
}  // namespace athena::backend

extern "C" {
void athena_fadd(void *a, size_t ca, void *b, size_t cb, void *c);
void athena_allocate(void *a, void *t);
void *athena_get_fast_pointer(void *a, void *t);
void athena_ffill(void *allocator, void *tensor, float f);
}

#endif  // ATHENA_RUNTIME_DRIVER_H
