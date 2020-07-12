/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "GraphPartitionPlanner.h"

using namespace athena::core;

namespace athena::backend::llvm {
std::vector<std::shared_ptr<Device>>
GraphPartitionPlanner::getPartitionedDevices(
    const std::vector<std::shared_ptr<Device>>& devices) {
  // todo abatashev: implement a more complex logic: partition by NUMA and
  // graph requirements
  mDevices = devices;
  return devices;
}
std::unordered_map<std::string_view, Device*>
GraphPartitionPlanner::getGraphPartitioning() {
  auto topology = mGraph.traverse();
  std::unordered_map<std::string_view, Device*> partitioning;

  auto ctx = mGraph.getContext();

  for (auto& cluster : topology.getClusters()) {
    for (auto& nodeState : cluster.content) {
      auto& node = ctx.internal()->getRef<internal::AbstractNodeInternal>(
          nodeState.nodeIndex);
      partitioning[node.getName().getString()] = mDevices[0].get();
    }
  }

  return partitioning;
}
} // namespace athena::backend::llvm
