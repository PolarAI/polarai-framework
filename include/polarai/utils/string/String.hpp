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

#pragma once

#include <polar_utils_export.h>
#include <polarai/utils/allocator/Allocator.hpp>

#include <cstddef>

namespace polarai::utils {
class POLAR_UTILS_EXPORT String {
public:
  String();
  String(const char* string, Allocator<byte> allocator = Allocator<byte>());
  String(const String&);
  String(String&&) noexcept;
  ~String();
  [[nodiscard]] const char* getString() const;
  [[nodiscard]] size_t getSize() const;

private:
  size_t mSize;
  Allocator<byte> mAllocator;
  const char* mData;
};
} // namespace polarai::utils
