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

#include <polar_core_export.h>
#include <polarai/core/PublicEntity.hpp>

namespace polarai::core {
namespace internal {
class AbstractNodeInternal;
}

/**
 * A Node represents a piece of computation in Graph
 */
class POLAR_CORE_EXPORT AbstractNode : public PublicEntity {
public:
  using InternalType = internal::AbstractNodeInternal;
};

} // namespace polarai::core
