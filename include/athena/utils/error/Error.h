/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_ERROR_H
#define ATHENA_ERROR_H

#include <athena/utils/Defines.h>
#include <athena/utils/Pointer.h>
#include <athena/utils/string/String.h>
#include <athena/utils/utils_export.h>

#include <memory>
#include <ostream>
#include <sstream>
#include <string_view>

namespace athena::utils {

/**
 * A non-fatal error
 */
class ATH_UTILS_EXPORT Error {
protected:
  int32_t mErrorCode;
  String mErrorMessage;

public:
  Error();
  template <typename... Args>
  explicit Error(int32_t errorCode, Args&&... messages);
  Error(const Error& error) = default;
  Error(Error&& error) noexcept = default;
  ~Error() = default;
  [[nodiscard]] const String& getMessage() const;

  explicit operator bool() const;
};

template <typename... Args> String mergeToString(Args&&... messages) {
  std::stringstream ss;
  (ss << ... << std::forward<Args>(messages));
  return String(ss.str().data());
}

template <typename... Args>
ATH_FORCE_INLINE Error::Error(int32_t errorCode, Args&&... messages)
    : mErrorCode(errorCode),
      mErrorMessage(mergeToString(std::forward<Args>(messages)...)) {}

ATH_UTILS_EXPORT ATH_FORCE_INLINE std::ostream& operator<<(std::ostream& stream,
                                                           const Error& err) {
  stream << err.getMessage().getString() << "\n";
  return stream;
}
} // namespace athena::utils

#endif // ATHENA_ERROR_H
