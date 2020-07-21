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

#include <windows.h>
#include <string>

namespace polarai::backend::generic {
std::unique_ptr<DynamicLibrary>
DynamicLibrary::create(std::string_view libName) {
  HINSTANCE libHandle = LoadLibrary(TEXT(libName.data()));

  return std::make_unique<DynamicLibrary>(static_cast<void*>(libHandle));
}

void* DynamicLibrary::lookup(std::string_view symbolName) {
  return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HINSTANCE>(mHandle), symbolName.data()));
}

auto DynamicLibrary::getLastError() -> std::string {
  return std::string("Not implemented");
}

DynamicLibrary::~DynamicLibrary() { FreeLibrary(reinterpret_cast<HINSTANCE>(mHandle)); }
} // namespace polarai::backend::llvm
