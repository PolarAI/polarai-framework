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

#ifndef ATHENA_OPERATIONINTERNAL_H
#define ATHENA_OPERATIONINTERNAL_H

#include <athena/core/Entity.h>
#include <athena/core/Generator.h>
#include <athena/core/core_export.h>
#include <athena/core/graph/Graph.h>
#include <athena/utils/string/StringView.h>

#include <tuple>
#include <vector>

namespace athena::core::internal {
class ATH_CORE_EXPORT OperationInternal : public Entity {
public:
  OperationInternal(utils::WeakPtr<ContextInternal> context,
                    utils::Index publicNodeIndex,
                    utils::String name = utils::String(""));

  ~OperationInternal() override = default;

  [[nodiscard]] virtual utils::Index
  createResultTensor(utils::SharedPtr<ContextInternal> context,
                     std::vector<TensorInternal*> tensorIndexes) const = 0;

  virtual void gen(utils::SharedPtr<ContextInternal> context,
                   core::internal::Generator& generator,
                   std::vector<utils::Index>& operationArguments) const = 0;

  virtual std::tuple<utils::Index, std::vector<core::internal::Edge>,
                     std::vector<utils::Index>>
  genDerivative(const core::NodeState* currentNodeState,
                size_t indexOfOutputDependence,
                utils::Index gradientGraphFinalNodeIndex) const = 0;

  [[nodiscard]] virtual size_t getOperandsCount() const = 0;
};
} // namespace athena::core::internal

#endif // ATHENA_OPERATIONIMPL_H
