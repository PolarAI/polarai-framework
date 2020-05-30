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
#include <athena/loaders/internal/CopyLoaderInternal.h>
#include <athena/operation/MulOperation.h>
#include <athena/operation/internal/MulOperationInternal.h>

namespace athena::operation::internal {
MulOperationInternal::MulOperationInternal(
    utils::SharedPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)) {}

utils::Index MulOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    std::vector<core::internal::TensorInternal*> tensorIndexes) const {
  auto dataType = tensorIndexes[0]->getDataType();
  auto tensorShape = tensorIndexes[0]->getShape();
  // TODO check preconditions
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), dataType, std::move(tensorShape));
}

core::internal::GenValue MulOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    core::internal::Generator& generator,
    std::vector<utils::Index>& operationArguments,
    core::internal::GenNode parentNode) const {
  // TODO call generator
  return core::internal::GenValue{};
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
MulOperationInternal::genDerivative(
    const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  // TODO might be replaced with just argument tensor index and edges{}
  // returning?
  auto context = mContext.lock();
  const auto& derivativeNodeDependence =
      currentNodeState->output[indexOfOutputDependence];
  const auto& argumentNode =
      context->getRef<core::internal::AbstractNodeInternal>(
          derivativeNodeDependence.nodeIndex);
  utils::Index sourceTensorIndex = -1;
  if (derivativeNodeDependence.mark == MulOperation::LEFT) {
    sourceTensorIndex =
        context
            ->getRef<core::internal::AbstractNodeInternal>(
                currentNodeState->input[MulOperation::RIGHT].nodeIndex)
            .getTensorIndex();
  } else {
    sourceTensorIndex =
        context
            ->getRef<core::internal::AbstractNodeInternal>(
                currentNodeState->input[MulOperation::LEFT].nodeIndex)
            .getTensorIndex();
  }
  auto copyLoaderIndex = context->create<loaders::internal::CopyLoaderInternal>(
      context, context->getNextPublicIndex(), sourceTensorIndex);
  auto nodeWithCopyOfSourceTensor =
      context->create<core::internal::InputNodeInternal>(
          context, context->getNextPublicIndex(),
          argumentNode.getTensorPtr()->getShape(),
          argumentNode.getTensorPtr()->getDataType(), true, copyLoaderIndex);
  auto hadamardOperationIndex = context->create<MulOperationInternal>(
      context, context->getNextPublicIndex());
  auto hadamardNode = context->create<core::internal::NodeInternal>(
      context, context->getNextPublicIndex(), hadamardOperationIndex);
  std::vector<core::internal::Edge> edges;
  edges.emplace_back(gradientGraphFinalNodeIndex, hadamardNode,
                     MulOperation::LEFT);
  edges.emplace_back(nodeWithCopyOfSourceTensor, hadamardNode,
                     MulOperation::RIGHT);
  std::vector<utils::Index> newInputNodes;
  newInputNodes.emplace_back(nodeWithCopyOfSourceTensor);
  return std::make_tuple(hadamardNode, edges, newInputNodes);
}

size_t MulOperationInternal::getOperandsCount() const { return 2; }
} // namespace athena::operation::internal
