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

#include <athena/core/AbstractLogger.h>
#include <athena/core/core_export.h>

#include <ostream>

namespace athena::core {
class ATH_CORE_EXPORT Logger : public AbstractLogger {
    private:
    std::ostream &mOutStream;

    public:
    explicit Logger(std::ostream &stream) : mOutStream(stream){};
    ~Logger() override = default;

    AbstractLogger &streamImpl(const std::string &data) override;
    AbstractLogger &streamImpl(const char *data) override;
    AbstractLogger &streamImpl(const std::string_view &data) override;
    AbstractLogger &streamImpl(const Error &data) override;
};
}  // namespace athena::core
#endif  // ATHENA_LOGGER_H
