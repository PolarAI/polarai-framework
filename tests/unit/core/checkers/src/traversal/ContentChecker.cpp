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
#include <traversal/ContentChecker.h>
#include <traversal/Utils.h>

namespace athena::tests::unit {
std::set<utils::Index> getSetOfNodes(const core::Cluster& cluster) {
  std::set<utils::Index> result;
  for (auto& nodeState : cluster.content) {
    result.insert(nodeState.nodeIndex);
  }
  return result;
}

bool checkTraversalContent(const core::Traversal& traversal,
                           const std::vector<std::set<utils::Index>>& target) {
  auto& clusters = traversal.getClusters();
  if (clusters.size() != target.size()) {
    std::cerr << "Clusters count doesn't equal to Target clusters count"
              << std::endl;
    std::cerr << "Clusters count: " << clusters.size() << std::endl;
    std::cerr << "Target clusters count: " << target.size() << std::endl;
    return false;
  }
  for (utils::Index index = 0; index < clusters.size(); ++index) {
    auto currentSetNodes = getSetOfNodes(clusters[index]);
    if (target[index] != currentSetNodes) {
      std::cerr << "Clusters aren't equal" << std::endl;
      showContainer(std::cerr, target[index], "Target");
      showContainer(std::cerr, currentSetNodes, "Cluster");
      return false;
    }
  }
  return true;
}
} // namespace athena::tests::unit