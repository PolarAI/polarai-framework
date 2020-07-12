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

#ifndef ATHENA_SIGMOIDOPERATIONINTERNAL_H
#define ATHENA_SIGMOIDOPERATIONINTERNAL_H

#include <athena/core/context/internal/ContextInternal.h>
#include <athena/core/operation/internal/OperationInternal.h>
#include <athena/utils/allocator/Allocator.h>
#include <polar_operation_export.h>

namespace athena::operation::internal {
class POLAR_OPERATION_EXPORT SigmoidOperationInternal
: public core::internal::OperationInternal {
public:
SigmoidOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
utils::Index publicNodeIndex, utils::String name = utils::String(""));

~SigmoidOperationInternal() override = default;

[[nodiscard]] utils::Index
createResultTensor(utils::SharedPtr<core::internal::ContextInternal> context,
                   const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
                   const std::vector<core::internal::TensorInternal*>& tensors)
const override;

core::internal::GenValue
gen(utils::SharedPtr<core::internal::ContextInternal> context,
    core::internal::Generator& generator,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors,
    const core::internal::TensorInternal* resultTensor,
    core::internal::GenNode parentNode) const override;

// output node and edges of generated graph
std::tuple<utils::Index, std::vector<core::internal::Edge>,
std::vector<utils::Index>>
genDerivative(const core::NodeState* inputNodeState,
              const core::NodeState* currentNodeState,
              size_t indexOfOutputDependence,
              utils::Index gradientGraphFinalNodeIndex) const override;

[[nodiscard]] size_t getOperandsCount() const override;
};
} // namespace athena::operation::internal

#endif // ATHENA_SIGMOIDOPERATIONINTERNAL_H
