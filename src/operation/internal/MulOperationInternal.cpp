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
#include <athena/operation/MulOperation.h>
#include <athena/operation/internal/MulOperationInternal.h>
#include <athena/loaders/DummyLoader.h>

using namespace athena::core::internal;

namespace athena::operation::internal {
MulOperationInternal::MulOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)) {}

utils::Index MulOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors) const {
  auto dataType = tensors[0]->getDataType();
  auto tensorShape = tensors[0]->getShape();
  // TODO check preconditions
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), dataType, std::move(tensorShape));
}

core::internal::GenValue MulOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    core::internal::Generator& generator,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors,
    const core::internal::TensorInternal* resultTensor,
    core::internal::GenNode parentNode) const {
  std::unordered_map<utils::Index, GenValue> argMap;
  GenValue a = parentNode.getOperand(mapMarkToLocalTensorIndex.at(MulOperation::LEFT));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(MulOperation::LEFT))->getPublicIndex()] = a;
  GenValue b = parentNode.getOperand(mapMarkToLocalTensorIndex.at(MulOperation::RIGHT));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(MulOperation::RIGHT))->getPublicIndex()] = b;

  std::unordered_map<utils::Index, GenValue> resultMap;
  GenValue out = parentNode.getResult();
  resultMap[resultTensor->getPublicIndex()] = out;

  GenValue size = generator.createConstant(static_cast<uint64_t>(tensors.at(0)->getShapeView().getTotalSize()));

  generator.setInsertionPoint(parentNode);

  lockTensors(generator, argMap, resultMap);

  GenValue res = generator.callBuiltin<builtin::Mul>(a, b, size, out);

  releaseTensors(generator, argMap, resultMap);

  return res;
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
MulOperationInternal::genDerivative(
    const core::NodeState* inputNodeState, const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  auto context = mContext.lock();
  const auto& derivativeNodeDependence =
      inputNodeState->output[indexOfOutputDependence];
  utils::Index sourceTensorIndex = -1;
  if (derivativeNodeDependence.mark == MulOperation::LEFT) {
    sourceTensorIndex =
        context
            ->getRef<core::internal::AbstractNodeInternal>(
                currentNodeState
                    ->findDependency(currentNodeState->input, MulOperation::RIGHT).nodeIndex)
            .getTensorIndex();
  } else {
    sourceTensorIndex =
        context
            ->getRef<core::internal::AbstractNodeInternal>(
                currentNodeState
                    ->findDependency(currentNodeState->input, MulOperation::LEFT).nodeIndex)
            .getTensorIndex();
  }
  auto& sourceTensor = context->getRef<core::internal::TensorInternal>(sourceTensorIndex);
  auto dummyLoaderIndex = context->create<loaders::internal::DummyLoaderInternal>(context, context->getNextPublicIndex(), (std::string("MulOp_DummyLoader") +
      std::to_string(context->getNextPublicIndex())).data());
  auto dummyNodeIndex =
      context->create<core::internal::InputNodeInternal>(context, context->getNextPublicIndex(),
                                                         sourceTensor.getShape(), sourceTensor.getDataType(),
                                                         true, dummyLoaderIndex, (std::string("MulOp_ArgHolder") +
              std::to_string(context->getNextPublicIndex())).data());
  auto& dummyNode = context->getRef<core::internal::AbstractNodeInternal>(dummyNodeIndex);
  dummyNode.setTensorIndex(sourceTensorIndex);
  auto mulOperationIndex = context->create<MulOperationInternal>(
      context, context->getNextPublicIndex(), (std::string("MulOp_FinalMulOperation") +
          std::to_string(context->getNextPublicIndex())).data());
  auto mulNode = context->create<core::internal::NodeInternal>(
      context, context->getNextPublicIndex(), mulOperationIndex, (std::string("MulOp_FinalMulNode") +
          std::to_string(context->getNextPublicIndex())).data());
  std::vector<core::internal::Edge> edges;
  edges.emplace_back(gradientGraphFinalNodeIndex, mulNode,
                     MulOperation::LEFT);
  edges.emplace_back(dummyNodeIndex, mulNode,
                     MulOperation::RIGHT);
  std::vector<utils::Index> newInputNodes;
  newInputNodes.emplace_back(dummyNodeIndex);
  return std::make_tuple(mulNode, edges, newInputNodes);
}

size_t MulOperationInternal::getOperandsCount() const { return 2; }
} // namespace athena::operation::internal
