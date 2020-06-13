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
#include <athena/loaders/internal/ConstantLoaderInternal.h>
#include <athena/operation/internal/AddOperationInternal.h>
#include <athena/operation/AddOperation.h>

using namespace athena::core::internal;

namespace athena::operation::internal {
AddOperationInternal::AddOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)) {}

utils::Index AddOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors) const {
  auto dataType = tensors[0]->getDataType();
  auto tensorShape = tensors[0]->getShape();
  // TODO check preconditions
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), dataType, std::move(tensorShape));
}

core::internal::GenValue AddOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    core::internal::Generator& generator,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors,
    const core::internal::TensorInternal* resultTensor,
    core::internal::GenNode parentNode) const {
  std::unordered_map<utils::Index, GenValue> argMap;
  GenValue a = parentNode.getOperand(mapMarkToLocalTensorIndex.at(AddOperation::LEFT));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(AddOperation::LEFT))->getPublicIndex()] = a;
  GenValue b = parentNode.getOperand(mapMarkToLocalTensorIndex.at(AddOperation::RIGHT));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(AddOperation::RIGHT))->getPublicIndex()] = b;

  std::unordered_map<utils::Index, GenValue> resultMap;
  GenValue out = parentNode.getResult();
  resultMap[resultTensor->getPublicIndex()] = out;

  generator.setInsertionPoint(parentNode);

  // TODO support other data types
  // TODO take scale values from operation
  GenValue scaleA = generator.createConstant(1.0f);
  GenValue scaleB = generator.createConstant(1.0f);
  GenValue size = generator.createConstant(static_cast<uint64_t>(tensors.at(0)->getShapeView().getTotalSize()));

  lockTensors(generator, argMap, resultMap);

  GenValue res = generator.callBuiltin<builtin::Add>(a, scaleA, b, scaleB, size, out);

  releaseTensors(generator, argMap, resultMap);
  return res;
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
AddOperationInternal::genDerivative(
    const core::NodeState* inputNodeState, const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  return std::make_tuple(gradientGraphFinalNodeIndex, std::vector<core::internal::Edge>{}, std::vector<utils::Index>{});
}

size_t AddOperationInternal::getOperandsCount() const { return 2; }
} // namespace athena::operation::internal
