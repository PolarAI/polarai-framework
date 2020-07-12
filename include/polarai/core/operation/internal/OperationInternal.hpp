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

#include <polarai/core/Entity.hpp>
#include <polarai/core/Generator.hpp>
#include <polarai/core/graph/Graph.hpp>
#include <polarai/utils/string/StringView.hpp>
#include <polar_core_export.h>

#include <tuple>
#include <vector>

namespace polarai::core::internal {
class POLAR_CORE_EXPORT OperationInternal : public Entity {
public:
  OperationInternal(utils::WeakPtr<ContextInternal> context,
                    utils::Index publicNodeIndex,
                    utils::String name = utils::String(""));

  ~OperationInternal() override = default;

  [[nodiscard]] virtual utils::Index createResultTensor(
      utils::SharedPtr<ContextInternal> context,
      const std::unordered_map<int64_t, utils::Index>&
          mapMarkToLocalTensorIndex,
      const std::vector<core::internal::TensorInternal*>& tensors) const = 0;

  virtual GenValue
  gen(utils::SharedPtr<ContextInternal> context,
      core::internal::Generator& generator,
      const std::unordered_map<int64_t, utils::Index>&
          mapMarkToLocalTensorIndex,
      const std::vector<core::internal::TensorInternal*>& tensors,
      const core::internal::TensorInternal* resultTensor,
      GenNode parentNode) const = 0;

  virtual std::tuple<utils::Index, std::vector<core::internal::Edge>,
                     std::vector<utils::Index>>
  genDerivative(const core::NodeState* inputNodeState,
                const core::NodeState* currentNodeState,
                size_t indexOfOutputDependence,
                utils::Index gradientGraphFinalNodeIndex) const = 0;

  [[nodiscard]] virtual size_t getOperandsCount() const = 0;

protected:
  void lockTensors(core::internal::Generator& generator,
                   std::unordered_map<utils::Index, GenValue> argMap,
                   std::unordered_map<utils::Index, GenValue> resultMap) const;
  void
  releaseTensors(core::internal::Generator& generator,
                 std::unordered_map<utils::Index, GenValue> argMap,
                 std::unordered_map<utils::Index, GenValue> resultMap) const;
};
} // namespace polarai::core::internal
