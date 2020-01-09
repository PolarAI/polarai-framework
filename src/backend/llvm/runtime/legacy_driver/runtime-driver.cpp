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

#include "runtime-driver.h"

#include "llvm/IR/Verifier.h"

#include <llvm/Target/TargetMachine.h>

namespace athena::backend::llvm {

LegacyRuntimeDriver::LegacyRuntimeDriver(::llvm::LLVMContext& ctx)
    : mLibraryHandle(nullptr), mContext(ctx) {}

LegacyRuntimeDriver::~LegacyRuntimeDriver() { unload(); }
LegacyRuntimeDriver&
LegacyRuntimeDriver::operator=(LegacyRuntimeDriver&& rhs) noexcept {
  unload();
  mLibraryHandle = rhs.mLibraryHandle;
  rhs.mLibraryHandle = nullptr;
  return *this;
}
void* LegacyRuntimeDriver::getFunctionPtr(std::string_view funcName) {
  if (void* function = dlsym(mLibraryHandle, funcName.data()); !function) {
    new ::athena::core::FatalError(core::ATH_FATAL_OTHER,
                                   "RuntimeDriver: " + std::string(dlerror()));
    return nullptr;
  } else {
    return function;
  }
}
void LegacyRuntimeDriver::load(std::string_view nameLibrary) {
  if (mLibraryHandle = dlopen(nameLibrary.data(), RTLD_LAZY); !mLibraryHandle) {
    new ::athena::core::FatalError(core::ATH_FATAL_OTHER,
                                   "RuntimeDriver: " + std::string(dlerror()));
  }
  prepareModules();
}
void LegacyRuntimeDriver::unload() {
  if (mLibraryHandle && dlclose(mLibraryHandle)) {
    ::athena::core::FatalError err(core::ATH_FATAL_OTHER,
                                   "RuntimeDriver: " + std::string(dlerror()));
  }
  mLibraryHandle = nullptr;
}
void LegacyRuntimeDriver::reload(std::string_view nameLibrary) {
  unload();
  load(nameLibrary);
}
bool LegacyRuntimeDriver::isLoaded() const { return mLibraryHandle != nullptr; }

void LegacyRuntimeDriver::prepareModules() {
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
    new core::FatalError(core::ATH_FATAL_OTHER, "incorrect ir");
  }
#endif
  mModules.push_back(std::move(newModule));
}
void LegacyRuntimeDriver::setProperAttrs(::llvm::Function* function) {
  function->addFnAttr(::llvm::Attribute::NoUnwind);
  function->addFnAttr(::llvm::Attribute::UWTable);
  function->addFnAttr(::llvm::Attribute::AlwaysInline);
}
DeviceContainer LegacyRuntimeDriver::getAvailableDevices() {
  auto devicesFunc =
      (DeviceContainer(*)())getFunctionPtr("getAvailableDevices");
  return devicesFunc();
}
void LegacyRuntimeDriver::initializeContext(DeviceContainer devices) {
  auto initCtxFunc =
      (void (*)(DeviceContainer))getFunctionPtr("initializeContext");
  initCtxFunc(devices);
}
void LegacyRuntimeDriver::releaseContext() {
  auto releaseCtxFunc = (void (*)())getFunctionPtr("releaseContext");
  releaseCtxFunc();
}
} // namespace athena::backend::llvm
