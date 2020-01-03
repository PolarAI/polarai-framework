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
DeviceContainer
GraphPartitionPlanner::getPartitionedDevices(DeviceContainer devices) {
  // todo abatashev: implement a more complex logic: partition by NUMA and
  // graph requirements
  mDevices = devices;
  return devices;
}
std::unordered_map<std::string_view, Device*>
GraphPartitionPlanner::getGraphPartitioning() {
  auto topology = mGraph.traverse();
  std::unordered_map<std::string_view, Device*> partitioning;

  auto& ctx = inner::getContext(mGraph);
  auto& nodeTable = inner::getNodeTable(ctx);

  for (auto cluster : topology.getClusters()) {
    for (auto& node : cluster.get<InputNode>()) {
      partitioning[nodeTable[node.nodeIndex]->getName()] = mDevices.devices;
    }
    for (auto& node : cluster.get<Node>()) {
      partitioning[nodeTable[node.nodeIndex]->getName()] = mDevices.devices;
    }
    for (auto& node : cluster.get<LossNode>()) {
      partitioning[nodeTable[node.nodeIndex]->getName()] = mDevices.devices;
    }
    for (auto& node : cluster.get<OutputNode>()) {
      partitioning[nodeTable[node.nodeIndex]->getName()] = mDevices.devices;
    }
  }

  return partitioning;
}
} // namespace athena::backend::llvm