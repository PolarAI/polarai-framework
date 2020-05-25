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

namespace athena::backend::llvm {
void* RuntimeDriver::getFunctionPtr(std::string_view funcName) {
  if (void* function = dlsym(mLibraryHandle, funcName.data()); !function) {
    new utils::FatalError(utils::ATH_FATAL_OTHER, "RuntimeDriver: ", dlerror());
    return nullptr;
  } else {
    return function;
  }
}
void* RuntimeDriver::loadOsLibrary(const char* fullPath, char** errMsg) {
  void* lib = dlopen(fullPath, RTLD_LAZY);
  *errMsg = dlerror();
  return lib;
}

void RuntimeDriver::closeOsLibrary(void* handle, char** errMsg) {
  dlclose(handle);
  *errMsg = dlerror();
}
} // namespace athena::backend::llvm