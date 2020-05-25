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

#ifndef ATHENA_ADDOPERATIONINTERNAL_H
#define ATHENA_ADDOPERATIONINTERNAL_H

#include <athena/core/context/internal/ContextInternal.h>
#include <athena/core/operation/internal/OperationInternal.h>
#include <athena/operation/operation_export.h>
#include <athena/utils/allocator/Allocator.h>

namespace athena::operation::internal {
class ATH_OPERATION_EXPORT AddOperationInternal
    : public core::internal::OperationInternal {
public:
  AddOperationInternal(
      utils::SharedPtr<core::internal::ContextInternal> context,
      utils::Index publicNodeIndex, utils::String name = utils::String(""));

  ~AddOperationInternal() override = default;

  [[nodiscard]] utils::Index
  createResultTensor(utils::SharedPtr<core::internal::ContextInternal> context,
                     std::vector<core::internal::TensorInternal*> tensorIndexes)
      const override;

  void gen(utils::SharedPtr<core::internal::ContextInternal> context,
           core::internal::Generator& generator,
           std::vector<utils::Index>& operationArguments) const override;

  // output node and edges of generated graph
  std::tuple<utils::Index, std::vector<core::internal::Edge>,
             std::vector<utils::Index>>
  genDerivative(const core::NodeState* currentNodeState,
                size_t indexOfOutputDependence,
                utils::Index gradientGraphFinalNodeIndex) const override;

  [[nodiscard]] virtual size_t getOperandsCount() const override;
};
} // namespace athena::operation::internal

#endif // ATHENA_ADDOPERATIONINTERNAL_H
