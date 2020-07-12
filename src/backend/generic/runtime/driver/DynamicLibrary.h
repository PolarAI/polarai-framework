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

#pragma once

#include <memory>
#include <string_view>
#include <string>

namespace athena::backend::llvm {
class DynamicLibrary {
public:
  static auto create(std::string_view libName)
      -> std::unique_ptr<DynamicLibrary>;
  auto lookup(std::string_view symbolName) -> void*;

  auto isValid() -> bool { return mHandle != nullptr; }

  auto getLastError() -> std::string;

  ~DynamicLibrary();

private:
  friend auto std::make_unique<DynamicLibrary>(void*&)
      -> std::unique_ptr<DynamicLibrary>;
  DynamicLibrary(void* handle) : mHandle(handle){};

  void* mHandle;
};
} // namespace athena::backend::llvm
