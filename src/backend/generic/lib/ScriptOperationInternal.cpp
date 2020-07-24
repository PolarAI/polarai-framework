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

#include "ScriptOperationInternal.hpp"
#include <polarai/backend/generic/ScriptOperation.hpp>
#include <polarai/core/node/internal/AbstractNodeInternal.hpp>
#include <polarai/core/node/internal/NodeInternal.hpp>
#include <polarai/loaders/internal/ConstantLoaderInternal.hpp>

using namespace polarai::core::internal;

namespace polarai::backend::generic::internal {
ScriptOperationInternal::ScriptOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, utils::String program, size_t numArgs,
    ShapeInferenceCallback scb, BuildDerivativeCallback dcb)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        "ScriptOperation"),
      mNumOperands(numArgs), mShapeInference(scb), mDerivativeCallback(dcb) {}

utils::Index ScriptOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors) const {
  return mShapeInference(context, mapMarkToLocalTensorIndex, tensors);
}

core::internal::GenValue ScriptOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    core::internal::Generator& generator,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors,
    const core::internal::TensorInternal* resultTensor,
    core::internal::GenNode parentNode) const {
  generator.setInsertionPoint(parentNode);

  // TODO implement

  return GenValue{};
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
ScriptOperationInternal::genDerivative(
    const core::NodeState* inputNodeState,
    const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  return mDerivativeCallback(inputNodeState, currentNodeState,
                             indexOfOutputDependence,
                             gradientGraphFinalNodeIndex);
}

size_t ScriptOperationInternal::getOperandsCount() const {
  return mNumOperands;
}
} // namespace polarai::backend::generic::internal
