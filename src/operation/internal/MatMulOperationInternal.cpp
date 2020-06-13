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
#include <athena/operation/MatMulOperation.h>
#include <athena/operation/MulOperation.h>
#include <athena/operation/MulConcatOperation.h>
#include <athena/operation/internal/MulOperationInternal.h>
#include <athena/core/tensor/internal/TensorInternal.h>
#include <athena/loaders/internal/ConstantLoaderInternal.h>
#include <athena/loaders/internal/DummyLoaderInternal.h>

using namespace athena::core::internal;

namespace athena::operation::internal {
MatMulOperationInternal::MatMulOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, bool isLeftTranspose, bool isRightTranspose, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)), mIsLeftTranspose(isLeftTranspose), mIsRightTranspose(isRightTranspose) {}

utils::Index MatMulOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors) const {
  auto dataType = tensors[0]->getDataType();
  auto& leftMatrixTensorShape = tensors[mapMarkToLocalTensorIndex.at(MatMulOperation::LEFT)]->getShape();
  auto& rightMatrixTensorShape = tensors[mapMarkToLocalTensorIndex.at(MatMulOperation::RIGHT)]->getShape();
  uint64_t m = mIsLeftTranspose ? leftMatrixTensorShape.getShape()[1] : leftMatrixTensorShape.getShape()[0];
  uint64_t n = mIsRightTranspose ? rightMatrixTensorShape.getShape()[0] : rightMatrixTensorShape.getShape()[1];

  // TODO check preconditions
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), dataType, core::TensorShape{m, n});
}

core::internal::GenValue MatMulOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    core::internal::Generator& generator,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors,
    const core::internal::TensorInternal* resultTensor,
    core::internal::GenNode parentNode) const {
  std::unordered_map<utils::Index, GenValue> argMap;
  GenValue left = parentNode.getOperand(mapMarkToLocalTensorIndex.at(MatMulOperation::LEFT));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(MatMulOperation::LEFT))->getPublicIndex()] = left;
  GenValue right = parentNode.getOperand(mapMarkToLocalTensorIndex.at(MatMulOperation::RIGHT));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(MatMulOperation::RIGHT))->getPublicIndex()] = right;

  std::unordered_map<utils::Index, GenValue> resultMap;
  GenValue out = parentNode.getResult();
  resultMap[resultTensor->getPublicIndex()] = out;

  auto leftTensorPtr = tensors[mapMarkToLocalTensorIndex.at(MatMulOperation::LEFT)];
  auto rightTensorPtr = tensors[mapMarkToLocalTensorIndex.at(MatMulOperation::RIGHT)];

  GenValue transLeft = generator.createConstant(static_cast<uint64_t>(mIsLeftTranspose));
  GenValue transRight = generator.createConstant(static_cast<uint64_t>(mIsRightTranspose));

  GenValue m = generator.createConstant(mIsLeftTranspose ? static_cast<uint64_t>(leftTensorPtr->getShape()[1]) : static_cast<uint64_t>(leftTensorPtr->getShape()[0]));
  GenValue k = generator.createConstant(mIsLeftTranspose ? static_cast<uint64_t>(leftTensorPtr->getShape()[0]) : static_cast<uint64_t>(leftTensorPtr->getShape()[1]));
  GenValue n = generator.createConstant(mIsRightTranspose ? static_cast<uint64_t>(rightTensorPtr->getShape()[0]) : static_cast<uint64_t>(rightTensorPtr->getShape()[1]));

  generator.setInsertionPoint(parentNode);

  lockTensors(generator, argMap, resultMap);

  GenValue res = generator.callBuiltin<builtin::MatMul>(transLeft, transRight, m, n, k, left, right, out);

  releaseTensors(generator, argMap, resultMap);

  return res;
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
MatMulOperationInternal::genDerivative(
    const core::NodeState* inputNodeState, const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  // TODO now only for NxN gemm
  auto context = mContext.lock();
  std::vector<utils::Index> newInputNodes;
  std::vector<core::internal::Edge> edges;
  utils::Index newFinalIndex = 0;
  utils::Index matMulNodeIndex = 0;
  auto rightMatrixTensorIndex =
      context
          ->getRef<core::internal::AbstractNodeInternal>(
              currentNodeState
                  ->findDependency(currentNodeState->input, MatMulOperation::RIGHT).nodeIndex)
          .getTensorIndex();
  auto leftMatrixTensorIndex =
      context
          ->getRef<core::internal::AbstractNodeInternal>(
              currentNodeState
                  ->findDependency(currentNodeState->input, MatMulOperation::LEFT).nodeIndex)
          .getTensorIndex();
  auto& rightMatrixTensor = context->getRef<TensorInternal>(rightMatrixTensorIndex);
  auto& leftMatrixTensor = context->getRef<TensorInternal>(leftMatrixTensorIndex);
  auto identityLoaderIndex =
      context->create<loaders::internal::ConstantLoaderInternal>(
          context, context->getNextPublicIndex(), 1.0,
          (std::string("MatMulOp_IdentityLoader") +
              std::to_string(context->getNextPublicIndex()))
              .data());
  auto dummyLoaderIndex = context->create<loaders::internal::DummyLoaderInternal>(context, context->getNextPublicIndex(), (std::string("MatMulOp_ArgDummyLoader") +
      std::to_string(context->getNextPublicIndex())).data());
  if (inputNodeState->output[indexOfOutputDependence].mark == MatMulOperation::LEFT) {
    auto dummyNodeIndex =
        context->create<core::internal::InputNodeInternal>(context, context->getNextPublicIndex(),
                                                           rightMatrixTensor.getShape(), rightMatrixTensor.getDataType(),
                                                           true, dummyLoaderIndex, (std::string("MatMulOp_ArgHolderNode") +
                std::to_string(context->getNextPublicIndex())).data());
    auto& dummyNode = context->getRef<core::internal::AbstractNodeInternal>(dummyNodeIndex);
    dummyNode.setTensorIndex(rightMatrixTensorIndex);
    newInputNodes.emplace_back(dummyNodeIndex);

    auto nodeWithIdentityTensor =
        context->create<core::internal::InputNodeInternal>(
            context, context->getNextPublicIndex(),
            core::TensorShape{leftMatrixTensor.getShape()[0], rightMatrixTensor.getShape()[1]},
            rightMatrixTensor.getDataType(), true, identityLoaderIndex,
            (std::string("MatMulOp_IdentityNode") +
                std::to_string(context->getNextPublicIndex()))
                .data());
    newInputNodes.emplace_back(nodeWithIdentityTensor);
    auto matMulOperationIndex = context->create<MatMulOperationInternal>(context, context->getNextPublicIndex(), false, true, (std::string("MatMulOp_MatMulOperationNode") +
        std::to_string(context->getNextPublicIndex())).data());
    matMulNodeIndex = context->create<NodeInternal>(context, context->getNextPublicIndex(), matMulOperationIndex, (std::string("MatMulOp_MatMulNodeLeft") +
        std::to_string(context->getNextPublicIndex())).data());
    edges.emplace_back(nodeWithIdentityTensor, matMulNodeIndex,
                       MatMulOperation::LEFT);
    edges.emplace_back(dummyNodeIndex, matMulNodeIndex,
                       MatMulOperation::RIGHT);
  } else if (inputNodeState->output[indexOfOutputDependence].mark == MatMulOperation::RIGHT) {
    auto dummyNodeIndex =
        context->create<core::internal::InputNodeInternal>(context, context->getNextPublicIndex(),
                                                           leftMatrixTensor.getShape(), leftMatrixTensor.getDataType(),
                                                           true, dummyLoaderIndex, (std::string("MatMulOp_ArgHolderNode") +
                std::to_string(context->getNextPublicIndex())).data());
    auto& dummyNode = context->getRef<core::internal::AbstractNodeInternal>(dummyNodeIndex);
    dummyNode.setTensorIndex(leftMatrixTensorIndex);
    newInputNodes.emplace_back(dummyNodeIndex);

    auto nodeWithIdentityTensor =
        context->create<core::internal::InputNodeInternal>(
            context, context->getNextPublicIndex(),
            core::TensorShape{leftMatrixTensor.getShape()[0], rightMatrixTensor.getShape()[1]},
            leftMatrixTensor.getDataType(), true, identityLoaderIndex,
            (std::string("MatMulOp_IdentityNode") +
                std::to_string(context->getNextPublicIndex()))
                .data());
    newInputNodes.emplace_back(nodeWithIdentityTensor);
    auto matMulOperationIndex = context->create<MatMulOperationInternal>(context, context->getNextPublicIndex(), true, false, (std::string("MatMulOp_MatMulOperationNode") +
        std::to_string(context->getNextPublicIndex())).data());
    matMulNodeIndex = context->create<NodeInternal>(context, context->getNextPublicIndex(), matMulOperationIndex, (std::string("MatMulOp_MatMulNodeRight") +
        std::to_string(context->getNextPublicIndex())).data());
    edges.emplace_back(nodeWithIdentityTensor, matMulNodeIndex,
                       MatMulOperation::RIGHT);
    edges.emplace_back(dummyNodeIndex, matMulNodeIndex,
                       MatMulOperation::LEFT);
  } else {
    //TODO error
  }
  auto mulConcatOperationIndex = context->create<MulConcatOperationInternal>(context, context->getNextPublicIndex(), (std::string("MatMulOp_FinalMulConcatOperation") +
      std::to_string(context->getNextPublicIndex()))
      .data());
  newFinalIndex = context->create<NodeInternal>(context, context->getNextPublicIndex(), mulConcatOperationIndex, (std::string("MatMulOp_FinalMulConcatNode") +
      std::to_string(context->getNextPublicIndex()))
      .data());
  edges.emplace_back(matMulNodeIndex, newFinalIndex,
                     MulConcatOperation::LOCAL_DERIVATIVE);
  edges.emplace_back(gradientGraphFinalNodeIndex, newFinalIndex,
                     MulConcatOperation::GRADIENT);

  return std::make_tuple(newFinalIndex, edges, newInputNodes);
}

size_t MatMulOperationInternal::getOperandsCount() const { return 2; }
} // namespace athena::operation::internal
