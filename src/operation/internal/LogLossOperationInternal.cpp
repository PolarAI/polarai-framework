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
#include <athena/loaders/internal/DummyLoaderInternal.h>
#include <athena/operation/MulOperation.h>
#include <athena/operation/internal/AddOperationInternal.h>
#include <athena/operation/CombineOperation.h>
#include <athena/operation/DivideOperation.h>
#include <athena/operation/internal/LogLossOperationInternal.h>
#include <athena/operation/LogLossOperation.h>

using namespace athena::core::internal;

namespace athena::operation::internal {
LogLossOperationInternal::LogLossOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, utils::String name)
    : OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)) {}

utils::Index LogLossOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors) const {
  // TODO check preconditions
  auto dataType = tensors[0]->getDataType();
  auto tensorShape = tensors[0]->getShape();
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), dataType, std::move(tensorShape));
}

core::internal::GenValue LogLossOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    Generator& generator,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors,
    const core::internal::TensorInternal* resultTensor,
    GenNode parentNode) const {
  std::unordered_map<utils::Index, GenValue> argMap;
  GenValue groundTruth = parentNode.getOperand(mapMarkToLocalTensorIndex.at(LogLossOperation::GROUND_TRUTH));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(LogLossOperation::GROUND_TRUTH))->getPublicIndex()] = groundTruth;
  GenValue predicted = parentNode.getOperand(mapMarkToLocalTensorIndex.at(LogLossOperation::PREDICTED));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(LogLossOperation::PREDICTED))->getPublicIndex()] = predicted;

  std::unordered_map<utils::Index, GenValue> resultMap;
  GenValue out = parentNode.getResult();
  resultMap[resultTensor->getPublicIndex()] = out;

  GenValue size = generator.createConstant(static_cast<uint64_t>(tensors.at(0)->getShapeView().getTotalSize()));

  generator.setInsertionPoint(parentNode);

  lockTensors(generator, argMap, resultMap);

  GenValue res = generator.callBuiltin<builtin::LogLoss>(predicted, groundTruth, size, out);

  releaseTensors(generator, argMap, resultMap);
  return res;
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
LogLossOperationInternal::genDerivative(
    const core::NodeState* inputNodeState, const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  auto context = mContext.lock();
  std::vector<utils::Index> newInputNodes;
  std::vector<core::internal::Edge> edges;
  // Get predicted and ground truth tensors
  auto groundTruthTensorIndex =
      context
          ->getRef<core::internal::AbstractNodeInternal>(
              currentNodeState
                  ->findDependency(currentNodeState->input, LogLossOperation::GROUND_TRUTH).nodeIndex)
          .getTensorIndex();
  auto& groundTruthTensor = context->getRef<TensorInternal>(groundTruthTensorIndex);

  auto predictedTensorIndex =
      context
          ->getRef<core::internal::AbstractNodeInternal>(
              currentNodeState
                  ->findDependency(currentNodeState->input, LogLossOperation::PREDICTED).nodeIndex)
          .getTensorIndex();
  auto& predictedTensor = context->getRef<TensorInternal>(predictedTensorIndex);

  auto dummyLoaderIndex = context->create<loaders::internal::DummyLoaderInternal>(context, context->getNextPublicIndex(), (std::string("LogLossOp_DummyLoader") +
      std::to_string(context->getNextPublicIndex()))
      .data());

  auto groundTruthNodeIndex =
      context->create<core::internal::InputNodeInternal>(context, context->getNextPublicIndex(),
                                                         groundTruthTensor.getShape(), groundTruthTensor.getDataType(),
                                                         true, dummyLoaderIndex, (std::string("LogLossOp_GroundTruthNode") +
              std::to_string(context->getNextPublicIndex())).data());
  auto& groundTruthNode = context->getRef<AbstractNodeInternal>(groundTruthNodeIndex);
  groundTruthNode.setTensorIndex(groundTruthTensorIndex);
  newInputNodes.emplace_back(groundTruthNodeIndex);

  auto predictedNodeIndex =
      context->create<core::internal::InputNodeInternal>(context, context->getNextPublicIndex(),
                                                         predictedTensor.getShape(), predictedTensor.getDataType(),
                                                         true, dummyLoaderIndex, (std::string("LogLossOp_PredictedNode") +
              std::to_string(context->getNextPublicIndex())).data());
  auto& predictedNode = context->getRef<AbstractNodeInternal>(predictedNodeIndex);
  predictedNode.setTensorIndex(predictedTensorIndex);
  newInputNodes.emplace_back(predictedNodeIndex);

  auto predictedNodeCopyIndex =
      context->create<core::internal::InputNodeInternal>(context, context->getNextPublicIndex(),
                                                         predictedTensor.getShape(), predictedTensor.getDataType(),
                                                         true, dummyLoaderIndex, (std::string("LogLossOp_PredictedNode") +
              std::to_string(context->getNextPublicIndex())).data());
  auto& predictedNodeCopy = context->getRef<AbstractNodeInternal>(predictedNodeCopyIndex);
  predictedNodeCopy.setTensorIndex(predictedTensorIndex);
  newInputNodes.emplace_back(predictedNodeCopyIndex);

  // Func : (a - y) / (a - a^2)
  // Numerator:
  auto combineOperationIndex = context->create<CombineOperationInternal>(context, context->getNextPublicIndex(), 1.0, -1.0, (std::string("LogLossOp_CombineOperation") +
      std::to_string(context->getNextPublicIndex())).data());
  auto combineNodeNumeratorIndex = context->create<NodeInternal>(context, context->getNextPublicIndex(), combineOperationIndex, (std::string("LogLossOp_CombineNodeNumerator") +
      std::to_string(context->getNextPublicIndex())).data());
  edges.emplace_back(predictedNodeIndex, combineNodeNumeratorIndex,
                     CombineOperation::ALPHA);
  edges.emplace_back(groundTruthNodeIndex, combineNodeNumeratorIndex,
                     CombineOperation::BETA);

  // Denominator:
  auto mulOperationIndex = context->create<MulOperationInternal>(context, context->getNextPublicIndex(), (std::string("LogLossOp_MulOperation") +
      std::to_string(context->getNextPublicIndex())).data());
  auto mulNodeIndex = context->create<NodeInternal>(context, context->getNextPublicIndex(), mulOperationIndex, (std::string("LogLossOp_MulNodeDenominator") +
      std::to_string(context->getNextPublicIndex())).data());
  edges.emplace_back(predictedNodeIndex, mulNodeIndex,
                     MulOperation::LEFT);
  edges.emplace_back(predictedNodeCopyIndex, mulNodeIndex,
                     MulOperation::RIGHT);
  auto combineNodeDenominatorIndex = context->create<NodeInternal>(context, context->getNextPublicIndex(), combineOperationIndex, (std::string("LogLossOp_CombineNodeDenominator") +
      std::to_string(context->getNextPublicIndex())).data());
  edges.emplace_back(predictedNodeIndex, combineNodeDenominatorIndex,
                     CombineOperation::ALPHA);
  edges.emplace_back(mulNodeIndex, combineNodeDenominatorIndex,
                     CombineOperation::BETA);

  // Division
  auto divideOperationIndex = context->create<DivideOperationInternal>(context, context->getNextPublicIndex(), (std::string("LogLossOp_DivideOperation") +
      std::to_string(context->getNextPublicIndex())).data());
  auto divideNodeIndex = context->create<NodeInternal>(context, context->getNextPublicIndex(), divideOperationIndex, (std::string("LogLossOp_DivideNode") +
      std::to_string(context->getNextPublicIndex())).data());
  edges.emplace_back(combineNodeNumeratorIndex, divideNodeIndex,
                     DivideOperation::NUMERATOR);
  edges.emplace_back(combineNodeDenominatorIndex, divideNodeIndex,
                     DivideOperation::DENOMINATOR);

  // Mul to final of gradient
  auto mulFinalNodeIndex = context->create<NodeInternal>(context, context->getNextPublicIndex(), mulOperationIndex, (std::string("LogLossOp_FinalMulNode") +
      std::to_string(context->getNextPublicIndex())).data());
  edges.emplace_back(divideNodeIndex, mulFinalNodeIndex,
                     MulOperation::LEFT);
  edges.emplace_back(gradientGraphFinalNodeIndex, mulFinalNodeIndex,
                     MulOperation::RIGHT);

  return {mulFinalNodeIndex, edges, newInputNodes};
}

size_t LogLossOperationInternal::getOperandsCount() const { return 2; }
} // namespace athena::operation::internal
