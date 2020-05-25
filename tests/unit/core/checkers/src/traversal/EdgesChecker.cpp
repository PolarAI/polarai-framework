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

#include <iostream>
#include <traversal/EdgesChecker.h>
#include <traversal/Utils.h>

namespace athena::tests::unit {
std::ostream& operator<<(std::ostream& stream, const Edge& edge) {
  stream << edge.startNode << ' ' << edge.endNode << ' ' << edge.mark
         << std::endl;
  return stream;
}

std::set<Edge> createEdgesFromTraversal(const core::Traversal& traversal) {
  std::set<Edge> edges;
  for (auto& cluster : traversal.getClusters()) {
    for (auto& nodeState : cluster.content) {
      for (auto& outputDependence : nodeState.output) {
        auto edge = Edge{nodeState.nodeIndex, outputDependence.nodeIndex,
                         outputDependence.mark};
        edges.emplace(edge);
      }
      for (auto& inputDependence : nodeState.input) {
        auto edge = Edge{inputDependence.nodeIndex, nodeState.nodeIndex,
                         inputDependence.mark};
        edges.emplace(edge);
      }
    }
  }
  return edges;
}

bool checkEdges(const core::Traversal& traversal, const std::set<Edge>& edges) {
  auto traversalEdges = createEdgesFromTraversal(traversal);
  if (edges.size() != traversalEdges.size()) {
    std::cerr << "Edges count doesn't equal to traversal edges count"
              << std::endl;
    std::cerr << "Edges count: " << edges.size() << std::endl;
    std::cerr << "Traversal edges count: " << traversalEdges.size()
              << std::endl;
    showContainer(std::cerr, edges, "Edges: ");
    showContainer(std::cerr, traversalEdges, "Traversed edges: ");
    return false;
  }
  if (edges != traversalEdges) {
    std::cerr << "Edges doesn't equal to traversal edges" << std::endl;
    showContainer(std::cerr, edges, "Edges: ");
    showContainer(std::cerr, traversalEdges, "Traversal edges: ");
    return false;
  }
  return true;
}
} // namespace athena::tests::unit
