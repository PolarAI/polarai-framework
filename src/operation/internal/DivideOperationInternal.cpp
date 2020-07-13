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

#include <athena/core/node/internal/AbstractNodeInternal.hpp>
#include <athena/core/node/internal/NodeInternal.hpp>
#include <athena/loaders/DummyLoader.hpp>
#include <athena/operation/DivideOperation.hpp>
#include <athena/operation/internal/DivideOperationInternal.hpp>

using namespace athena::core::internal;

namespace athena::operation::internal {
DivideOperationInternal::DivideOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)) {}

utils::Index DivideOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors) const {
  auto dataType = tensors[0]->getDataType();
  auto tensorShape = tensors[0]->getShape();
  // TODO check preconditions
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), dataType, std::move(tensorShape));
}

core::internal::GenValue DivideOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    core::internal::Generator& generator,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors,
    const core::internal::TensorInternal* resultTensor,
    core::internal::GenNode parentNode) const {
  generator.setInsertionPoint(parentNode);

  std::unordered_map<utils::Index, GenValue> argMap;
  GenValue numerator = parentNode.getOperand(
      mapMarkToLocalTensorIndex.at(DivideOperation::NUMERATOR));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(DivideOperation::NUMERATOR))
             ->getPublicIndex()] = numerator;
  GenValue denominator = parentNode.getOperand(
      mapMarkToLocalTensorIndex.at(DivideOperation::DENOMINATOR));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(DivideOperation::DENOMINATOR))
             ->getPublicIndex()] = denominator;

  std::unordered_map<utils::Index, GenValue> resultMap;
  GenValue out = parentNode.getResult();
  resultMap[resultTensor->getPublicIndex()] = out;

  lockTensors(generator, argMap, resultMap);

  generator.callBuiltin<builtin::Divide>(numerator, denominator, out);

  releaseTensors(generator, argMap, resultMap);

  return out;
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
DivideOperationInternal::genDerivative(
    const core::NodeState* inputNodeState,
    const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  // TODO
  return {};
}

size_t DivideOperationInternal::getOperandsCount() const { return 2; }
} // namespace athena::operation::internal
