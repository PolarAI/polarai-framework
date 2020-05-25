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
#include <athena/operation/internal/AddOperationInternal.h>
#include <athena/operation/internal/MulOperationInternal.h>

namespace athena::operation::internal {
AddOperationInternal::AddOperationInternal(
    utils::SharedPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)) {}

utils::Index AddOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    std::vector<core::internal::TensorInternal*> tensorIndexes) const {
  auto dataType = tensorIndexes[0]->getDataType();
  auto tensorShape = tensorIndexes[0]->getShape();
  // TODO check preconditions
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), dataType, std::move(tensorShape));
}

void AddOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    core::internal::Generator& generator,
    std::vector<utils::Index>& operationArguments) const {
  // TODO call generator
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
AddOperationInternal::genDerivative(
    const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  // TODO might be replaced with just currentNodeIndex and edges{} returning?
  auto context = mContext.lock();
  const auto& derivativeNodeDependence =
      currentNodeState->output[indexOfOutputDependence];
  const auto& argumentNode =
      context->getRef<core::internal::AbstractNodeInternal>(
          derivativeNodeDependence.nodeIndex);
  auto constantLoaderIndex =
      context->create<loaders::internal::ConstantLoaderInternal>(
          context, context->getNextPublicIndex(), 1.0,
          (std::string("AddOp_IdentityLoader") +
           std::to_string(context->getNextPublicIndex()))
              .data());
  auto nodeWithIdentityTensor =
      context->create<core::internal::InputNodeInternal>(
          context, context->getNextPublicIndex(),
          argumentNode.getTensorPtr()->getShape(),
          argumentNode.getTensorPtr()->getDataType(), true, constantLoaderIndex,
          (std::string("AddOp_IdentityNode") +
           std::to_string(context->getNextPublicIndex()))
              .data());
  auto mulOperationIndex = context->create<MulOperationInternal>(
      context, context->getNextPublicIndex(),
      (std::string("AddOp_MulOperation") +
       std::to_string(context->getNextPublicIndex()))
          .data());
  auto mulNodeIndex = context->create<core::internal::NodeInternal>(
      context, context->getNextPublicIndex(), mulOperationIndex,
      (std::string("AddOp_MulNode") +
       std::to_string(context->getNextPublicIndex()))
          .data());
  std::vector<core::internal::Edge> edges;
  edges.emplace_back(gradientGraphFinalNodeIndex, mulNodeIndex,
                     MulOperation::LEFT);
  edges.emplace_back(nodeWithIdentityTensor, mulNodeIndex, MulOperation::RIGHT);
  std::vector<utils::Index> newInputNodes;
  newInputNodes.emplace_back(nodeWithIdentityTensor);
  return std::make_tuple(mulNodeIndex, edges, newInputNodes);
}

size_t AddOperationInternal::getOperandsCount() const { return 2; }
} // namespace athena::operation::internal
