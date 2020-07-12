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

#ifndef ATHENA_OPERATION_H
#define ATHENA_OPERATION_H

#include <polar_core_export.h>

namespace athena::core {
namespace internal {
class OperationInternal;
}
/**
 * Operation is an abstract computation, like addition or multiplication
 */
class POLAR_CORE_EXPORT Operation {
public:
  enum Arguments { Unmarked };
  using InternalType = internal::OperationInternal;
  virtual ~Operation() = default;
};
} // namespace athena::core

#endif // ATHENA_OPERATION_H
