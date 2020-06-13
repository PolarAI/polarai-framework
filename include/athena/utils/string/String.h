/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_STRING_H
#define ATHENA_STRING_H

#include <athena/utils/allocator/Allocator.h>
#include <athena/utils/utils_export.h>

#include <cstddef>

namespace athena::utils {
class ATH_UTILS_EXPORT String {
public:
  String();
  String(const char* string, Allocator allocator = Allocator());
  String(const String&);
  String(String&&) noexcept;
  ~String();
  [[nodiscard]] const char* getString() const;
  [[nodiscard]] size_t getSize() const;

private:
  size_t mSize;
  Allocator mAllocator;
  const char* mData;
};
} // namespace athena::utils

#endif // ATHENA_STRING_H
