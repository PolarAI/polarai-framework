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

#ifndef ATHENA_FATALERROR_H
#define ATHENA_FATALERROR_H

#include "athena/utils/logger/Logger.h"

#include <athena/core/core_export.h>
#include <athena/utils/error/Error.h>
#include <athena/utils/logger/log.h>

#include <csignal>
#include <iostream>
#include <regex>
#include <string_view>
#ifndef WIN32
#include <cxxabi.h>
#include <execinfo.h>
#include <unistd.h>
#endif

namespace athena::utils {

enum FatalErrorType : int32_t {
  ATH_FATAL_OTHER = 1,
  ATH_NOT_IMPLEMENTED = 2,
  ATH_BAD_CAST = 3,
  ATH_BAD_ACCESS = 4,
  ATH_ASSERT = 5
};

namespace {
void print_stacktrace() {
#ifndef WIN32
  void* array[10];
  size_t size;

  size = backtrace(array, 10);

  char** symbolList = backtrace_symbols(array, size);

  for (size_t i = 1; i < size; i++) {
#ifdef __clang__
    std::smatch funcNameMatch;
    std::regex mangledFuncName(R"(\S+(?=\s\+))");
    std::string line(symbolList[i]);
    std::regex_search(line, funcNameMatch, mangledFuncName);

    std::smatch addrMatch;
    std::regex addrRegEx("0x[0-9a-f]+");
    std::regex_search(line, addrMatch, addrRegEx);

    std::string functionName = funcNameMatch.str(0);
    std::string addr = addrMatch.str(0);

    err() << std::to_string(i).data() << ". ";
    err() << addr.data() << "\t";
    int status;
    err() << abi::__cxa_demangle(functionName.c_str(), nullptr, nullptr,
                                 &status);
    err() << "\n";
#else
    // todo implement demangling for GCC
    err() << i << "." << symbolList[i] << "\n";
#endif
  }

#endif
}
} // namespace

/**
 * A fatal error. Creating instances of this class forces program to stop.
 */
class ATH_UTILS_EXPORT FatalError : public Error {
public:
  template <typename... Args>
  explicit FatalError(FatalErrorType errorCode, Args... messages);
};
template <typename... Args>
ATH_UTILS_EXPORT FatalError::FatalError(FatalErrorType errorCode,
                                        Args... messages)
    : Error(errorCode, messages...) {
  err() << mErrorMessage.getString() << "\n";

#ifdef DEBUG
  print_stacktrace();
  std::raise(SIGABRT);
#else
  exit(errorCode);
#endif
}
} // namespace athena::utils

namespace athena::utils {
template <typename... Args>
UniquePtr<FatalError> athena_assert(bool assert, Args&&... messages) {
#if defined(DEBUG) || defined(ATHENA_ASSERT)
  if (!assert) {
    return makeUnique<FatalError>(FatalErrorType::ATH_ASSERT,
                                  std::forward<Args>(messages)...);
  }
#endif
  return nullptr;
}
} // namespace athena::utils

#endif // ATHENA_FATALERROR_H
