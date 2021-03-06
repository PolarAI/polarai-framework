//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include "DynamicLibrary.hpp"

#include <dlfcn.h>
#include <string>

namespace polarai::backend::generic {
std::unique_ptr<DynamicLibrary>
DynamicLibrary::create(std::string_view libName) {
  void* libHandle = dlopen(libName.data(), RTLD_LAZY | RTLD_LOCAL);

  return std::make_unique<DynamicLibrary>(libHandle);
}

void* DynamicLibrary::lookup(std::string_view symbolName) {
  return dlsym(mHandle, symbolName.data());
}

auto DynamicLibrary::getLastError() -> std::string {
  return std::string(dlerror());
}

DynamicLibrary::~DynamicLibrary() { dlclose(mHandle); }
} // namespace polarai::backend::llvm
