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
#include <polarai/utils/error/Error.hpp>

#include <string>
#include <string_view>

namespace polarai::utils {

class Error;
/**
 * Abstract Athena logger interface
 */
class POLAR_UTILS_EXPORT AbstractLogger {
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

} // namespace polarai::utils
