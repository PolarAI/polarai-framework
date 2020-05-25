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

#ifndef ATHENA_TRAVERSALCLUSTERSEDGESCHECKER_H
#define ATHENA_TRAVERSALCLUSTERSEDGESCHECKER_H

#include <athena/core/graph/Traversal.h>
#include <athena/utils/Index.h>
#include <set>

namespace athena::tests::unit {
struct Edge {
  utils::Index startNode{};
  utils::Index endNode{};
  size_t mark{};
  Edge(utils::Index startNode, utils::Index endNode, size_t mark)
      : startNode(startNode), endNode(endNode), mark(mark) {}
  bool operator==(const Edge& rhs) const {
    return startNode == rhs.startNode && endNode == rhs.endNode &&
           mark == rhs.mark;
  }
  bool operator<(const Edge& rhs) const { return startNode < rhs.startNode; }
};

std::ostream& operator<<(std::ostream& stream, const Edge& edge);

bool checkEdges(const core::Traversal& traversal, const std::set<Edge>& edges);
} // namespace athena::tests::unit

#endif // ATHENA_TRAVERSALCLUSTERSEDGESCHECKER_H
