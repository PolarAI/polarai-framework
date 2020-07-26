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

#include <polar_backend_generic_export.h>
#include <polarai/core/operation/Operation.hpp>

namespace polarai::backend::generic {
namespace internal {
class ScriptOperationInternal;
}
class POLAR_BACKEND_GENERIC_EXPORT ScriptOperation : public core::Operation {
public:
  using InternalType = internal::ScriptOperationInternal;
  enum Arguments {
    ARG0 = 100,
    ARG1,
    ARG2,
    ARG3,
    ARG4,
    ARG5,
    ARG6,
    ARG7,
    ARG8,
    ARG9
  };
};
} // namespace polarai::backend::generic
