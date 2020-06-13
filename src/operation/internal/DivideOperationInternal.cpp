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

#include <athena/core/node/internal/AbstractNodeInternal.h>
#include <athena/core/node/internal/NodeInternal.h>
#include <athena/operation/DivideOperation.h>
#include <athena/operation/internal/DivideOperationInternal.h>
#include <athena/loaders/DummyLoader.h>

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
  std::unordered_map<utils::Index, GenValue> argMap;
  GenValue numerator = parentNode.getOperand(mapMarkToLocalTensorIndex.at(DivideOperation::NUMERATOR));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(DivideOperation::NUMERATOR))->getPublicIndex()] = numerator;
  GenValue denominator = parentNode.getOperand(mapMarkToLocalTensorIndex.at(DivideOperation::DENOMINATOR));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(DivideOperation::DENOMINATOR))->getPublicIndex()] = denominator;

  std::unordered_map<utils::Index, GenValue> resultMap;
  GenValue out = parentNode.getResult();
  resultMap[resultTensor->getPublicIndex()] = out;

  GenValue size = generator.createConstant(static_cast<uint64_t>(tensors.at(0)->getShapeView().getTotalSize()));

  generator.setInsertionPoint(parentNode);

  lockTensors(generator, argMap, resultMap);

  GenValue res = generator.callBuiltin<builtin::Divide>(numerator, denominator, size, out);

  releaseTensors(generator, argMap, resultMap);

  return res;
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
DivideOperationInternal::genDerivative(
    const core::NodeState* inputNodeState, const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  // TODO
  return {};
}

size_t DivideOperationInternal::getOperandsCount() const { return 2; }
} // namespace athena::operation::internal
