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

#include <polarai/core/graph/EdgeMark.hpp>
#include <polarai/utils/Index.hpp>
#include <polar_core_export.h>

namespace polarai::core::internal {
struct POLAR_CORE_EXPORT Edge {
  size_t startNodeIndex;
  size_t endNodeIndex;
  EdgeMark mark;
  Edge(const Edge& rhs) = default;
  Edge(Edge&& rhs) = default;
  Edge(utils::Index startNodeIndex, utils::Index endNodeIndex, EdgeMark mark)
      : startNodeIndex(startNodeIndex), endNodeIndex(endNodeIndex), mark(mark) {
  }
  Edge& operator=(const Edge& rhs) = default;
  Edge& operator=(Edge&& rhs) = default;
  bool operator==(const Edge& rhs) const {
    return startNodeIndex == rhs.startNodeIndex &&
           endNodeIndex == rhs.endNodeIndex;
  }
  bool operator<(const Edge& rhs) const {
    return startNodeIndex < rhs.startNodeIndex;
  }
};
} // namespace polarai::core::internal
