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

#ifndef ATHENA_EDGE_H
#define ATHENA_EDGE_H

#include <athena/core/core_export.h>
#include <athena/core/graph/EdgeMark.h>
#include <athena/utils/Index.h>

namespace athena::core::internal {
struct ATH_CORE_EXPORT Edge {
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
} // namespace athena::core::internal

#endif // ATHENA_EDGE_H
