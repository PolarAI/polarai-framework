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
#include <polarai/utils/string/String.hpp>

namespace polarai::utils {
class POLAR_UTILS_EXPORT StringView {
public:
  explicit StringView(const String& string);
  ~StringView() = default;
  [[nodiscard]] const char* getString() const;
  [[nodiscard]] size_t getSize() const;

private:
  const String* mString;
};
} // namespace polarai::utils
