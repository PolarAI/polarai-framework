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

#ifndef ATHENA_ABSTRACTLOGER_H
#define ATHENA_ABSTRACTLOGER_H

#include <athena/core/core_export.h>
#include <athena/utils/error/Error.h>

#include <string>
#include <string_view>

namespace athena::utils {

class Error;
/**
 * Abstract Athena logger interface
 */
class ATH_UTILS_EXPORT AbstractLogger {
public:
  AbstractLogger() = default;
  AbstractLogger(const AbstractLogger&) = default;
  AbstractLogger(AbstractLogger&&) noexcept = default;
  AbstractLogger& operator=(const AbstractLogger&) = default;
  AbstractLogger& operator=(AbstractLogger&&) noexcept = default;
  virtual ~AbstractLogger() = default;

  template <typename Type> AbstractLogger& operator<<(Type&& data) {
    return streamImpl(std::forward<Type>(data));
  }

protected:
  virtual AbstractLogger& streamImpl(const char* data) = 0;
  virtual AbstractLogger& streamImpl(size_t data) = 0;
};

} // namespace athena::utils
#endif // ATHENA_ABSTRACTLOGER_H
