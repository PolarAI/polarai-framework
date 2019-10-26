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

#include <athena/backend/llvm/runtime-driver/runtime-driver.h>

#include "llvm/IR/Verifier.h"

#include <llvm/Support/Debug.h>
#include <llvm/Target/TargetMachine.h>

namespace athena::backend::llvm {

RuntimeDriver::RuntimeDriver(::llvm::LLVMContext &ctx)
    : mLibraryHandle(nullptr), mContext(ctx) {}

RuntimeDriver::~RuntimeDriver() {
    unload();
}
RuntimeDriver &RuntimeDriver::operator=(RuntimeDriver &&rhs) noexcept {
    unload();
    mLibraryHandle = rhs.mLibraryHandle;
    rhs.mLibraryHandle = nullptr;
    return *this;
}
void *RuntimeDriver::getFunctionPtr(std::string_view funcName) {
    if (void *function = dlsym(mLibraryHandle, funcName.data()); !function) {
        new ::athena::core::FatalError(
            1, "RuntimeDriver: " + std::string(dlerror()));
        return nullptr;
    } else {
        return function;
    }
}
void RuntimeDriver::load(std::string_view nameLibrary) {
    if (mLibraryHandle = dlopen(nameLibrary.data(), RTLD_LAZY);
        !mLibraryHandle) {
        new ::athena::core::FatalError(
            1, "RuntimeDriver: " + std::string(dlerror()));
    }
    prepareModules();
}
void RuntimeDriver::unload() {
    if (mLibraryHandle && dlclose(mLibraryHandle)) {
        ::athena::core::FatalError err(
            1, "RuntimeDriver: " + std::string(dlerror()));
    }
    mLibraryHandle = nullptr;
}
void RuntimeDriver::reload(std::string_view nameLibrary) {
    unload();
    load(nameLibrary);
}
bool RuntimeDriver::isLoaded() const {
    return mLibraryHandle != nullptr;
}

void RuntimeDriver::prepareModules() {
    auto newModule = std::make_unique<::llvm::Module>("runtime", mContext);
    newModule->setTargetTriple(::llvm::sys::getDefaultTargetTriple());
    ::llvm::IRBuilder<> builder(mContext);
    generateLLVMIrBindings(mContext, *newModule, builder);
#ifdef DEBUG
    bool brokenDebugInfo = false;
    std::string str;
    ::llvm::raw_string_ostream stream(str);
    bool isBroken = ::llvm::verifyModule(*newModule, &stream, &brokenDebugInfo);
    stream.flush();
    if (isBroken || brokenDebugInfo) {
        err() << str;
        newModule->print(::llvm::errs(), nullptr);
        new core::FatalError(10, "incorrect ir");
    }
#endif
    mModules.push_back(std::move(newModule));
}
void RuntimeDriver::setProperAttrs(::llvm::Function *function) {
    function->addFnAttr(::llvm::Attribute::NoUnwind);
    function->addFnAttr(::llvm::Attribute::UWTable);
    function->addFnAttr(::llvm::Attribute::AlwaysInline);
}
}  // namespace athena::backend::llvm
