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

#include <athena/core/core_export.h>

#include <ostream>
#include <sstream>
#include <string_view>

namespace athena::core {

/**
 * A non-fatal error
 */
class ATH_CORE_EXPORT Error {
protected:
  int32_t mErrorCode;
  std::string mErrorMessage;

public:
  Error();
  template <typename... Args>
  explicit Error(int32_t errorCode, Args&&... messages);
  Error(const Error& error) = default;
  Error(Error&& error) noexcept = default;
  Error& operator=(const Error& error) = default;
  Error& operator=(Error&& error) noexcept = default;
  ~Error() = default;

  operator bool() const;
  friend std::ostream& operator<<(std::ostream& stream, const Error& err);
};
template <typename... Args>
ATH_CORE_EXPORT Error::Error(int32_t errorCode, Args&&... messages)
    : mErrorCode(errorCode) {
  std::stringstream ss;
  (ss << ... << std::forward<Args>(messages));
  mErrorMessage = ss.str();
}
} // namespace athena::core

#endif // ATHENA_ERROR_H
