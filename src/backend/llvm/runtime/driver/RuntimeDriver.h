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

#ifndef ATHENA_RUNTIMEDRIVER_H
#define ATHENA_RUNTIMEDRIVER_H

#include <athena/backend/llvm/driver/driver_export.h>
#include <athena/backend/llvm/runtime/Device.h>
#include <athena/core/FatalError.h>
#include <dlfcn.h>
#include <string>
#include <string_view>

namespace athena::backend::llvm {
/// The Runtime Driver manages available Athena runtimes.
class ATH_RT_LLVM_DRIVER_EXPORT RuntimeDriver {
private:
  using HasFeatureT = bool (*)(const char*);
  using HasBuiltinT = bool (*)(const char*, const char*);

  std::vector<void*> mLinkableModules;
  void* mLibraryHandle{};
  HasFeatureT mHasFeatureHandle = nullptr;
  bool (*mHasBuiltinHandle)(const char*, const char*) = nullptr;

  static std::string resolveFinalLibraryName(std::string_view);

  static std::string getFullPathName(const std::string& libraryName);

  void* getFunctionPtr(std::string_view funcName);

  static void* loadOsLibrary(const char* fullPath, char** errMsg);

  static void closeOsLibrary(void*, char** errMsg);

public:
  RuntimeDriver() = default;
  RuntimeDriver(const RuntimeDriver& rhs) = delete;
  RuntimeDriver(RuntimeDriver&& rhs) noexcept = default;
  ~RuntimeDriver();

  RuntimeDriver& operator=(const RuntimeDriver& rhs) = delete;
  RuntimeDriver& operator=(RuntimeDriver&& rhs) noexcept;

  void load(std::string_view libraryName = "");
  void unload();
  void reload(std::string_view libraryName = "");
  [[nodiscard]] bool isLoaded() const;

  DeviceContainer getAvailableDevices();
  void initializeContext(DeviceContainer devices);
  void releaseContext();

  std::vector<void*>& getLinkableModules() { return mLinkableModules; }

  bool hasFeature(const char* featureName) {
    return mHasFeatureHandle(featureName);
  }

  bool hasBuiltin(const char* builtinName, const char* typeName) {
    return mHasBuiltinHandle(builtinName, typeName);
  }
};
} // namespace athena::backend::llvm

#endif // ATHENA_RUNTIMEDRIVER_H
