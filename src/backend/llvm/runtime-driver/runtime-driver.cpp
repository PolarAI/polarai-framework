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
#include <llvm/Support/Debug.h>


namespace athena::backend::llvm {

RuntimeDriver kRuntimeDriver;

RuntimeDriver::RuntimeDriver() : mLibraryHandle(nullptr) {}
RuntimeDriver::RuntimeDriver(std::string_view nameLibrary) {
    load(nameLibrary);
}
RuntimeDriver::~RuntimeDriver() {
    unload();
}
RuntimeDriver& RuntimeDriver::operator=(RuntimeDriver&& rhs) noexcept {
    unload();
    mLibraryHandle = rhs.mLibraryHandle;
    mFaddPointer = rhs.mFaddPointer;
    rhs.mLibraryHandle = nullptr;
    rhs.mFaddPointer = nullptr;
    return *this;
}
void* RuntimeDriver::getFunction(std::string_view nameFunction) {
    if (void* function = dlsym(mLibraryHandle, nameFunction.data());
        !function) {
        ::athena::core::FatalError(1,
                                   "RuntimeDriver: " + std::string(dlerror()));
        return nullptr;
    } else {
        return function;
    }
}
void RuntimeDriver::load(std::string_view nameLibrary) {
    if (mLibraryHandle = dlopen(nameLibrary.data(), RTLD_LAZY);
        !mLibraryHandle) {
        ::athena::core::FatalError(1,
                                   "RuntimeDriver: " + std::string(dlerror()));
    }
    mFaddPointer =
        reinterpret_cast<void (*)(void*, size_t, void*, size_t, void*)>(
            getFunction("athena_fadd"));

    mAllocatePointer = reinterpret_cast<void (*)(void*, void*)>(
        getFunction("athena_allocate"));

    mGetFPPointer = reinterpret_cast<void* (*)(void*, void*)>(
        getFunction("athena_get_fast_pointer"));

    mFfillPointer = reinterpret_cast<void (*)(void*, void*, float)>(
        getFunction("athena_ffill"));
}
void RuntimeDriver::unload() {
    if (mLibraryHandle && dlclose(mLibraryHandle)) {
        ::athena::core::FatalError(1,
                                   "RuntimeDriver: " + std::string(dlerror()));
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
::llvm::ArrayRef<::llvm::Value *> RuntimeDriver::getArgs(::llvm::Function *function) {
    std::vector<::llvm::Value *> argValues;

    for (auto &arg : function->args()) {
        argValues.push_back(&arg);
    }

    return argValues;
}
void RuntimeDriver::prepareModules(::llvm::LLVMContext &ctx) {
    auto newModule = std::make_unique<::llvm::Module>("id", ctx);
    ::llvm::IRBuilder<> builder(ctx);
    generateLLLVMIrBindings(ctx, *newModule, builder);
    newModule->print(::llvm::dbgs(), nullptr);

}
}  // namespace athena::backend

extern "C" {
//void athena_fadd(void* a, size_t ca, void* b, size_t cb, void* c) {
//    athena::backend::kRuntimeDriver.athena_fadd(a, ca, b, cb, c);
//}
//void athena_allocate(void* a, void* t) {
//    athena::backend::kRuntimeDriver.athena_allocate(a, t);
//}
//void* athena_get_fast_pointer(void* a, void* t) {
//    return athena::backend::kRuntimeDriver.athena_get_fast_pointer(a, t);
//}
//void athena_ffill(void* allocator, void* tensor, float f) {
//    athena::backend::kRuntimeDriver.athena_ffill(allocator, tensor, f);
//}
}
