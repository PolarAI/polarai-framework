/*
 * Copyright (c) 2018 Athena. All rights reserved.
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

#ifndef ATHENA_LOGLOSSOPERATION_H
#define ATHENA_LOGLOSSOPERATION_H

#include <athena/core/operation/Operation.h>
#include <athena/operation/internal/LogLossOperationInternal.h>
#include <athena/operation/operation_export.h>

namespace athena::operation {
class ATH_OPERATION_EXPORT LogLossOperation : public core::Operation {
public:
using InternalType = internal::LogLossOperationInternal;
enum Arguments { GROUND_TRUTH=30, PREDICTED };
};
} // namespace athena::operation

#endif // ATHENA_LOGLOSSOPERATION_H
