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

#include <polarai/backend/generic/runtime/Device.hpp>
#include <polarai/core/graph/Graph.hpp>

namespace polarai::backend::generic {
class GraphPartitionPlanner {
private:
  core::Graph& mGraph;
  std::vector<std::shared_ptr<Device>> mDevices;

public:
  explicit GraphPartitionPlanner(core::Graph& graph) : mGraph(graph){};
  std::vector<std::shared_ptr<Device>>
  getPartitionedDevices(const std::vector<std::shared_ptr<Device>>& devices);
  std::unordered_map<std::string_view, Device*> getGraphPartitioning();
};
} // namespace polarai::backend::llvm
