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
#include <polarai/utils/logger/AbstractLogger.hpp>

#include <ostream>

namespace polarai::utils {
#ifdef DEBUG
namespace internal {
POLAR_UTILS_EXPORT void debugLoggerFatalError();
}
#endif
class Logger : public AbstractLogger {
public:
  explicit Logger(std::ostream& stream) : mOutStream(&stream){};
  ~Logger() override = default;

  AbstractLogger& streamImpl(const char* data) override {
    return streamImplInternal(data);
  }

  AbstractLogger& streamImpl(size_t data) override {
    return streamImplInternal(data);
  }

protected:
  template <typename Type> AbstractLogger& streamImplInternal(Type data) {
#ifdef DEBUG
    if (!mOutStream) {
      internal::debugLoggerFatalError();
    }
#endif
    *mOutStream << data;
    return *this;
  }

private:
  std::ostream* mOutStream;
};
} // namespace polarai::utils
