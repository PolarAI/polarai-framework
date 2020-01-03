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

#ifndef ATHENA_GRAPHPARTITIONPLANNER_H
#define ATHENA_GRAPHPARTITIONPLANNER_H

#include <athena/backend/llvm/runtime/Device.h>
#include <athena/core/Graph.h>

namespace athena::backend::llvm {
class GraphPartitionPlanner {
private:
  core::Graph& mGraph;
  DeviceContainer mDevices;

public:
  explicit GraphPartitionPlanner(core::Graph& graph) : mGraph(graph){};
  DeviceContainer getPartitionedDevices(DeviceContainer devices);
  std::unordered_map<std::string_view, Device*> getGraphPartitioning();
};
} // namespace athena::backend::llvm

#endif // ATHENA_GRAPHPARTITIONPLANNER_H
