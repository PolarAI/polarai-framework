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

#ifndef ATHENA_STRINGVIEW_H
#define ATHENA_STRINGVIEW_H

#include <athena/utils/string/String.h>

namespace athena::utils {
class ATH_UTILS_EXPORT StringView {
public:
  explicit StringView(const String& string);
  ~StringView() = default;
  [[nodiscard]] const char* getString() const;
  [[nodiscard]] size_t getSize() const;

private:
  const String* mString;
};
} // namespace athena::utils

#endif // ATHENA_STRINGVIEW_H
