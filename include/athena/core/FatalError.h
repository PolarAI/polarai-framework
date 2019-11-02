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

#include "Logger.h"

#include <athena/core/Error.h>
#include <athena/core/core_export.h>
#include <athena/core/log.h>

#include <csignal>
#include <iostream>
#include <string_view>

namespace athena::core {

/**
 * A fatal error. Creating instances of this class forces program to stop.
 */
class ATH_CORE_EXPORT FatalError : public Error {
    public:
    template <typename... Args>
    explicit FatalError(int32_t errorCode, Args... messages);
};
template <typename... Args>
FatalError::FatalError(int32_t errorCode, Args... messages)
    : Error(errorCode, messages...) {
    err() << mErrorMessage;
#ifdef DEBUG
    std::raise(SIGABRT);
#else
    exit(errorCode);
#endif
}
}  // namespace athena::core

#endif  // ATHENA_FATALERROR_H
