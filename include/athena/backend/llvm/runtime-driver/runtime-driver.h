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

    void *getFunctionPtr(std::string_view funcName);

    std::vector<std::unique_ptr<::llvm::Module>> mModules;

    void generateLLVMIrBindings(::llvm::LLVMContext &ctx,
                                ::llvm::Module &module,
                                ::llvm::IRBuilder<> &builder);

    static void setProperAttrs(::llvm::Function *function);

    void prepareModules();

    ::llvm::LLVMContext &mContext;

public:
    explicit RuntimeDriver(::llvm::LLVMContext &ctx);
    RuntimeDriver(const RuntimeDriver &rhs) = delete;
    RuntimeDriver(RuntimeDriver &&rhs) noexcept = default;
    ~RuntimeDriver();

    RuntimeDriver &operator=(const RuntimeDriver &rhs) = delete;
    RuntimeDriver &operator=(RuntimeDriver &&rhs) noexcept;

    void load(std::string_view nameLibrary);
    void unload();
    void reload(std::string_view nameLibrary);
    bool isLoaded() const;

    std::vector<std::unique_ptr<::llvm::Module>> &getModules() {
        // todo clone modules
        return mModules;
    };
};
}  // namespace athena::backend


#endif  // ATHENA_RUNTIME_DRIVER_H
