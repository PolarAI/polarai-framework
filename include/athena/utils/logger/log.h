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

#ifndef ATHENA_LOG_H
#define ATHENA_LOG_H

#include <athena/core/core_export.h>
#include <athena/utils/Pointer.h>
#include <athena/utils/logger/AbstractLogger.h>
#include <athena/utils/logger/Logger.h>

#include <iostream>
#include <memory>

namespace athena::utils {
class ATH_UTILS_EXPORT LogHolder {
  UniquePtr<AbstractLogger> mLog;
  UniquePtr<AbstractLogger> mErr;

  template <typename LoggerType, typename... Args>
  void setStream(UniquePtr<AbstractLogger>& stream, Args&&... args) {
    stream.reset(new LoggerType(std::forward<Args>(args)...));
  }

public:
  LogHolder()
      : mLog(makeUnique<Logger>(std::cout)),
        mErr(makeUnique<Logger>(std::cerr)) {
    std::cout << "Log holder was initialized" << std::endl;
  }
  ~LogHolder() = default;
  LogHolder(const LogHolder& rhs) = delete;
  LogHolder(LogHolder&& rhs) noexcept = delete;

  LogHolder& operator=(const LogHolder& rhs) = delete;
  LogHolder& operator=(LogHolder&& rhs) noexcept = delete;

  template <typename LoggerType, typename... Args>
  friend void setLogStream(Args&&... args);
  template <typename LoggerType, typename... Args>
  friend void setErrStream(Args&&... args);
  friend AbstractLogger& log();
  friend AbstractLogger& err();
};

extern const ATH_UTILS_EXPORT utils::UniquePtr<LogHolder> logHolder;

template <typename LoggerType, typename... Args>
ATH_UTILS_EXPORT void setLogStream(Args&&... args) {
  logHolder->setStream<LoggerType>(logHolder->mLog,
                                   std::forward<Args>(args)...);
}
template <typename LoggerType, typename... Args>
ATH_UTILS_EXPORT void setErrStream(Args&&... args) {
  logHolder->setStream<LoggerType>(logHolder->mErr,
                                   std::forward<Args>(args)...);
}
ATH_UTILS_EXPORT AbstractLogger& log();
ATH_UTILS_EXPORT AbstractLogger& err();
} // namespace athena::utils

#endif // ATHENA_LOG_H
