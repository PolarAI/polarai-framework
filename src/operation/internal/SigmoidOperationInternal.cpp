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
#include <athena/operation/SigmoidOperation.h>
#include <athena/operation/CombineOperation.h>
#include <athena/loaders/DummyLoader.h>
#include <athena/loaders/ConstantLoader.h>

using namespace athena::core::internal;

namespace athena::operation::internal {
SigmoidOperationInternal::SigmoidOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)) {}

utils::Index SigmoidOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors) const {
  auto dataType = tensors[0]->getDataType();
  auto tensorShape = tensors[0]->getShape();
  // TODO check preconditions
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), dataType, std::move(tensorShape));
}

core::internal::GenValue SigmoidOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    core::internal::Generator& generator,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors,
    const core::internal::TensorInternal* resultTensor,
    core::internal::GenNode parentNode) const {
  std::unordered_map<utils::Index, GenValue> argMap;
  GenValue input = parentNode.getOperand(0);
  argMap[tensors.at(0)->getPublicIndex()] = input;

  std::unordered_map<utils::Index, GenValue> resultMap;
  GenValue out = parentNode.getResult();
  resultMap[resultTensor->getPublicIndex()] = out;

  GenValue size = generator.createConstant(static_cast<uint64_t>(tensors.at(0)->getShapeView().getTotalSize()));

  generator.setInsertionPoint(parentNode);

  lockTensors(generator, argMap, resultMap);

  GenValue res = generator.callBuiltin<builtin::Sigmoid>(input, size, out);

  releaseTensors(generator, argMap, resultMap);

  return res;
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
SigmoidOperationInternal::genDerivative(
    const core::NodeState* inputNodeState, const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  auto context = mContext.lock();
  std::vector<utils::Index> newInputNodes;
  std::vector<core::internal::Edge> edges;

  auto sigmoidTensorIndex =
      context->getRef<core::internal::AbstractNodeInternal>(
          currentNodeState->nodeIndex).getTensorIndex();
  auto& sigmoidTensor = context->getRef<TensorInternal>(sigmoidTensorIndex);
  auto dummyLoaderIndex = context->create<loaders::internal::DummyLoaderInternal>(context, context->getNextPublicIndex());
  auto dummyNodeIndex =
      context->create<core::internal::InputNodeInternal>(context, context->getNextPublicIndex(),
                                                         sigmoidTensor.getShape(), sigmoidTensor.getDataType(),
                                                         true, dummyLoaderIndex, (std::string("SigmoidOp_DummySigmoidValueHolder") +
              std::to_string(context->getNextPublicIndex()))
                                                             .data());
  auto& dummyNode = context->getRef<core::internal::AbstractNodeInternal>(dummyNodeIndex);
  dummyNode.setTensorIndex(sigmoidTensorIndex);
  newInputNodes.emplace_back(dummyNodeIndex);

  auto identityLoaderIndex =
      context->create<loaders::internal::ConstantLoaderInternal>(
          context, context->getNextPublicIndex(), 1.0,
          (std::string("SigmoidOp_IdentityLoader") +
              std::to_string(context->getNextPublicIndex()))
              .data());
  auto nodeWithIdentityTensor =
      context->create<core::internal::InputNodeInternal>(
          context, context->getNextPublicIndex(),
          sigmoidTensor.getShape(),
          sigmoidTensor.getDataType(), true, identityLoaderIndex,
          (std::string("SigmoidOp_IdentityNode") +
              std::to_string(context->getNextPublicIndex()))
              .data());
  newInputNodes.emplace_back(nodeWithIdentityTensor);

  auto combineOperationIndex = context->create<CombineOperationInternal>(context, context->getNextPublicIndex(), 1.0, -1.0, (std::string("SigmoidOp_CombineOperation") +
      std::to_string(context->getNextPublicIndex()))
      .data());
  auto combineNodeIndex = context->create<NodeInternal>(context, context->getNextPublicIndex(), combineOperationIndex, (std::string("SigmoidOp_CombineNode") +
      std::to_string(context->getNextPublicIndex()))
      .data());
  edges.emplace_back(nodeWithIdentityTensor, combineNodeIndex,
                     CombineOperation::ALPHA);
  edges.emplace_back(dummyNodeIndex, combineNodeIndex,
                     CombineOperation::BETA);

  auto mulOperationIndex = context->create<MulOperationInternal>(context, context->getNextPublicIndex(), (std::string("SigmoidOp_MulOperation") +
      std::to_string(context->getNextPublicIndex()))
      .data());
  auto mulNodeIndex = context->create<NodeInternal>(context, context->getNextPublicIndex(), mulOperationIndex, (std::string("SigmoidOp_MulNode") +
      std::to_string(context->getNextPublicIndex()))
      .data());
  edges.emplace_back(combineNodeIndex, mulNodeIndex,
                     MulOperation::LEFT);
  edges.emplace_back(dummyNodeIndex, mulNodeIndex,
                     MulOperation::RIGHT);

  auto finalNodeIndex = context->create<NodeInternal>(context, context->getNextPublicIndex(), mulOperationIndex, (std::string("SigmoidOp_FinalMulNode") +
      std::to_string(context->getNextPublicIndex()))
      .data());
  edges.emplace_back(gradientGraphFinalNodeIndex, finalNodeIndex,
                     MulOperation::LEFT);
  edges.emplace_back(mulNodeIndex, finalNodeIndex,
                     MulOperation::RIGHT);

  return std::make_tuple(finalNodeIndex, edges, newInputNodes);
}

size_t SigmoidOperationInternal::getOperandsCount() const { return 2; }
} // namespace athena::operation::internal
