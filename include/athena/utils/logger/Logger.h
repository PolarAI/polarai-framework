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
#ifndef ATHENA_LOGGER_H
#define ATHENA_LOGGER_H

#include <athena/core/core_export.h>
#include <athena/utils/logger/AbstractLogger.h>

#include <ostream>

namespace athena::utils {
#ifdef DEBUG
namespace internal {
ATH_UTILS_EXPORT void debugLoggerFatalError();
}
#endif
class ATH_UTILS_EXPORT Logger : public AbstractLogger {
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
} // namespace athena::utils
#endif // ATHENA_LOGGER_H
