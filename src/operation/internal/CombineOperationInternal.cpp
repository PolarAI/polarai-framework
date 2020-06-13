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
#include <athena/operation/MulOperation.h>
#include <athena/operation/internal/CombineOperationInternal.h>
#include <athena/operation/internal/MulOperationInternal.h>
#include <athena/operation/CombineOperation.h>

using namespace athena::core::internal;

namespace athena::operation::internal {
CombineOperationInternal::CombineOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, float alpha, float beta, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)), mAlpha(alpha), mBeta(beta) {}

utils::Index CombineOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors) const {
  auto dataType = tensors[0]->getDataType();
  auto tensorShape = tensors[0]->getShape();
  // TODO check preconditions
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), dataType, std::move(tensorShape));
}

core::internal::GenValue CombineOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    core::internal::Generator& generator,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors,
    const core::internal::TensorInternal* resultTensor,
    core::internal::GenNode parentNode) const {
  std::unordered_map<utils::Index, GenValue> argMap;
  GenValue a = parentNode.getOperand(mapMarkToLocalTensorIndex.at(CombineOperation::ALPHA));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(CombineOperation::ALPHA))->getPublicIndex()] = a;
  GenValue b = parentNode.getOperand(mapMarkToLocalTensorIndex.at(CombineOperation::BETA));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(CombineOperation::BETA))->getPublicIndex()] = b;

  std::unordered_map<utils::Index, GenValue> resultMap;
  GenValue out = parentNode.getResult();
  resultMap[resultTensor->getPublicIndex()] = out;

  GenValue size = generator.createConstant(static_cast<uint64_t>(tensors.at(0)->getShapeView().getTotalSize()));

  generator.setInsertionPoint(parentNode);

  GenValue scaleA = generator.createConstant(mAlpha);
  GenValue scaleB = generator.createConstant(mBeta);

  lockTensors(generator, argMap, resultMap);

  GenValue res = generator.callBuiltin<builtin::Add>(a, scaleA, b, scaleB, size, out);

  releaseTensors(generator, argMap, resultMap);

  return res;
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
CombineOperationInternal::genDerivative(
    const core::NodeState* inputNodeState, const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  auto context = mContext.lock();
  const auto& derivativeNodeDependence =
      inputNodeState->output[indexOfOutputDependence];
  const auto& combineNode =
      context->getRef<core::internal::AbstractNodeInternal>(
          derivativeNodeDependence.nodeIndex);
  float constant = 0;
  if (derivativeNodeDependence.mark == CombineOperation::ALPHA) {
    constant = mAlpha;
  } else {
    constant = mBeta;
  }
  auto constantLoaderIndex =
      context->create<loaders::internal::ConstantLoaderInternal>(
          context, context->getNextPublicIndex(), constant,
          (std::string("CombOp_IdentityLoader") +
           std::to_string(context->getNextPublicIndex()))
              .data());
  auto nodeWithConstantTensor =
      context->create<core::internal::InputNodeInternal>(
          context, context->getNextPublicIndex(),
          combineNode.getTensorPtr()->getShape(),
          combineNode.getTensorPtr()->getDataType(), true, constantLoaderIndex,
          (std::string("CombOp_IdentityNode") +
           std::to_string(context->getNextPublicIndex()))
              .data());
  auto mulOperationIndex = context->create<MulOperationInternal>(
      context, context->getNextPublicIndex(),
      (std::string("CombOp_FinalMulOperation") +
       std::to_string(context->getNextPublicIndex()))
          .data());
  auto mulNodeIndex = context->create<core::internal::NodeInternal>(
      context, context->getNextPublicIndex(), mulOperationIndex,
      (std::string("CombOp_FinalMulNode") +
       std::to_string(context->getNextPublicIndex()))
          .data());
  std::vector<core::internal::Edge> edges;
  edges.emplace_back(gradientGraphFinalNodeIndex, mulNodeIndex,
                     MulOperation::LEFT);
  edges.emplace_back(nodeWithConstantTensor, mulNodeIndex, MulOperation::RIGHT);
  std::vector<utils::Index> newInputNodes;
  newInputNodes.emplace_back(nodeWithConstantTensor);
  return std::make_tuple(mulNodeIndex, edges, newInputNodes);
}

size_t CombineOperationInternal::getOperandsCount() const { return 2; }
} // namespace athena::operation::internal
