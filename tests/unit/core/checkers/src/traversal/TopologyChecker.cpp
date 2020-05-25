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
#include <traversal/TopologyChecker.h>

namespace athena::tests::unit {
size_t getIndexOfNodeState(const std::vector<core::Dependency>& dependence,
                           utils::Index nodeIndex) {
  for (size_t index = 0; index < dependence.size(); ++index) {
    if (dependence[index].nodeIndex == nodeIndex) {
      return index;
    }
  }
  return -1;
}

bool checkNodeState(const core::Traversal& traversal,
                    const core::NodeState& nodeState) {
  auto dependence = nodeState.input;
  dependence.insert(dependence.end(), nodeState.output.begin(),
                    nodeState.output.end());
  for (auto& elementDependence : dependence) {
    auto remoteNodeIndex =
        traversal.getClusters()[elementDependence.clusterIndex]
            .content[elementDependence.nodeStateIndex]
            .nodeIndex;
    auto localNodeIndex = elementDependence.nodeIndex;
    if (remoteNodeIndex != localNodeIndex) {
      std::cerr << "Input dependence case: Node index that was got by topology "
                   "doesn't equal to local node index."
                << std::endl;
      std::cerr << "Local node index: " << localNodeIndex << std::endl;
      std::cerr << "Remote node index: " << remoteNodeIndex << std::endl;
      return false;
    }
  }
  for (auto& inputDependence : nodeState.input) {
    auto& remoteVector = traversal.getClusters()[inputDependence.clusterIndex]
                             .content[inputDependence.nodeStateIndex]
                             .output;
    auto found_iterator =
        getIndexOfNodeState(remoteVector, nodeState.nodeIndex);
    if (found_iterator == static_cast<size_t>(-1)) {
      std::cerr << "Input dependence doesn't contain required node index"
                << std::endl;
      std::cerr << "Node index: " << nodeState.nodeIndex << std::endl;
      return false;
    }
  }
  for (auto& outputDependence : nodeState.output) {
    auto& remoteVector = traversal.getClusters()[outputDependence.clusterIndex]
                             .content[outputDependence.nodeStateIndex]
                             .input;
    auto found_iterator =
        getIndexOfNodeState(remoteVector, nodeState.nodeIndex);
    if (found_iterator == static_cast<size_t>(-1)) {
      std::cerr << "Output dependence doesn't contain required node index"
                << std::endl;
      std::cerr << "Node index: " << nodeState.nodeIndex << std::endl;
      return false;
    }
  }
  return true;
}

bool checkTopology(const core::Traversal& traversal) {
  auto& clusters = traversal.getClusters();
  for (auto& cluster : clusters) {
    for (auto& nodeState : cluster.content) {
      if (!checkNodeState(traversal, nodeState)) {
        return false;
      }
    }
  }
  return true;
}
} // namespace athena::tests::unit
