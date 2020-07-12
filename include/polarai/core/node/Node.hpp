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

#include <athena/core/node/AbstractNode.hpp>
#include <athena/core/node/internal/NodeInternal.hpp>
#include <polar_core_export.h>

namespace athena::core {
/**
 * A Node represents a piece of data loading to graph.
 */
class POLAR_CORE_EXPORT Node {
public:
  using InternalType = internal::NodeInternal;
};
} // namespace athena::core
