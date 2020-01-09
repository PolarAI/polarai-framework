//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include "RuntimeDriver.h"

#define LIB_PREFIX "lib"
#ifdef __APPLE__
#define LIB_POSTFIX ".dylib"
#elif defined(__linux__)
#define LIB_POSTFIX ".so"
#endif

namespace athena::backend::llvm {

RuntimeDriver::~RuntimeDriver() { unload(); }

RuntimeDriver& RuntimeDriver::operator=(RuntimeDriver&& rhs) noexcept {
  unload();
  mLibraryHandle = rhs.mLibraryHandle;
  rhs.mLibraryHandle = nullptr;
  return *this;
}

void RuntimeDriver::load(std::string_view libraryName) {
  std::string name = resolveFinalLibraryName(libraryName);
  char* errMsg = nullptr;
  if (mLibraryHandle = loadOsLibrary(name.data(), &errMsg);
      !mLibraryHandle || errMsg) {
    new ::athena::core::FatalError(core::ATH_FATAL_OTHER,
                                   "RuntimeDriver: ", errMsg);
  }

  mHasBuiltinHandle =
      reinterpret_cast<HasBuiltinT>(getFunctionPtr("hasBuiltin"));
  mHasFeatureHandle =
      reinterpret_cast<HasFeatureT>(getFunctionPtr("hasFeature"));

  mLinkableModules.push_back(mLibraryHandle);
}

void RuntimeDriver::unload() {
  if (mLibraryHandle) {
    char* errMsg;
    closeOsLibrary(mLibraryHandle, &errMsg);
    if (errMsg) {
      ::athena::core::FatalError err(
          core::ATH_FATAL_OTHER, "RuntimeDriver: " + std::string(dlerror()));
    }
  }
  mLibraryHandle = nullptr;
  mHasBuiltinHandle = nullptr;
  mHasFeatureHandle = nullptr;
  mLinkableModules.clear();
}

void RuntimeDriver::reload(std::string_view nameLibrary) {
  unload();
  load(nameLibrary);
}

bool RuntimeDriver::isLoaded() const { return mLibraryHandle != nullptr; }

DeviceContainer RuntimeDriver::getAvailableDevices() {
  auto devicesFunc =
      (DeviceContainer(*)())getFunctionPtr("getAvailableDevices");
  return devicesFunc();
}

void RuntimeDriver::initializeContext(DeviceContainer devices) {
  auto initCtxFunc =
      (void (*)(DeviceContainer))getFunctionPtr("initializeContext");
  initCtxFunc(devices);
}

void RuntimeDriver::releaseContext() {
  auto releaseCtxFunc = (void (*)())getFunctionPtr("releaseContext");
  releaseCtxFunc();
}

std::string
RuntimeDriver::resolveFinalLibraryName(std::string_view libraryName) {
  if (libraryName != "") {
    return getFullPathName(libraryName.data());
  }
  if (auto envRt = getenv("ATHENA_RUNTIME"); envRt) {
    return getFullPathName(envRt);
  }
  return getFullPathName(std::string(LIB_PREFIX) + "AthenaLLVMCpuRuntime" +
                         std::string(LIB_POSTFIX));
}

std::string RuntimeDriver::getFullPathName(const std::string& libraryName) {
  // todo full-featured implementation
  return libraryName;
}
} // namespace athena::backend::llvm
