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

#ifndef ATHENA_COMBINETIONOPERATIONINTERNAL_H
#define ATHENA_COMBINETIONOPERATIONINTERNAL_H

#include <athena/core/context/internal/ContextInternal.h>
#include <athena/core/operation/internal/OperationInternal.h>
#include <athena/operation/operation_export.h>
#include <athena/utils/allocator/Allocator.h>

namespace athena::operation::internal {
class ATH_OPERATION_EXPORT CombineOperationInternal
: public core::internal::OperationInternal {
public:
  CombineOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
utils::Index publicNodeIndex, float alpha, float beta, utils::String name = utils::String(""));

~CombineOperationInternal() override = default;

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

private:
  float mAlpha;
  float mBeta;
};
} // namespace athena::operation::internal

#endif // ATHENA_COMBINETIONOPERATIONINTERNAL_H
