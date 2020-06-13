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
#include <athena/operation/MulConcatOperation.h>
#include <athena/operation/internal/MulConcatOperationInternal.h>
#include <athena/loaders/DummyLoader.h>

using namespace athena::core::internal;

namespace athena::operation::internal {
MulConcatOperationInternal::MulConcatOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)) {}

utils::Index MulConcatOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors) const {
  auto dataType = tensors[0]->getDataType();
  auto tensorShape = tensors[mapMarkToLocalTensorIndex.at(MulConcatOperation::LOCAL_DERIVATIVE)]->getShape();
  // TODO check preconditions
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), dataType, std::move(tensorShape));
}

core::internal::GenValue MulConcatOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    core::internal::Generator& generator,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors,
    const core::internal::TensorInternal* resultTensor,
    core::internal::GenNode parentNode) const {
  std::unordered_map<utils::Index, GenValue> argMap;
  GenValue gradient = parentNode.getOperand(mapMarkToLocalTensorIndex.at(MulConcatOperation::GRADIENT));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(MulConcatOperation::GRADIENT))->getPublicIndex()] = gradient;
  GenValue localDerivative = parentNode.getOperand(mapMarkToLocalTensorIndex.at(MulConcatOperation::LOCAL_DERIVATIVE));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(MulConcatOperation::LOCAL_DERIVATIVE))->getPublicIndex()] = localDerivative;

  std::unordered_map<utils::Index, GenValue> resultMap;
  GenValue out = parentNode.getResult();
  resultMap[resultTensor->getPublicIndex()] = out;

  GenValue gradientSize = generator.createConstant(static_cast<uint64_t>(tensors[mapMarkToLocalTensorIndex.at(MulConcatOperation::GRADIENT)]->getShapeView().getTotalSize()));
  GenValue localDerivativeSize = generator.createConstant(static_cast<uint64_t>(tensors[mapMarkToLocalTensorIndex.at(MulConcatOperation::LOCAL_DERIVATIVE)]->getShapeView().getTotalSize()));

  generator.setInsertionPoint(parentNode);

  lockTensors(generator, argMap, resultMap);

  GenValue res = generator.callBuiltin<builtin::MulConcat>(gradient, gradientSize, localDerivative, localDerivativeSize, out);

  releaseTensors(generator, argMap, resultMap);

  return res;
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
MulConcatOperationInternal::genDerivative(
    const core::NodeState* inputNodeState, const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  //TODO
  return {};
}

size_t MulConcatOperationInternal::getOperandsCount() const { return 2; }
} // namespace athena::operation::internal
