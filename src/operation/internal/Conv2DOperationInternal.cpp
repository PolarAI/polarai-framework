//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include <polarai/core/node/internal/AbstractNodeInternal.hpp>
#include <polarai/core/node/internal/NodeInternal.hpp>
#include <polarai/loaders/DummyLoader.hpp>
#include <polarai/operation/Conv2DOperation.hpp>
#include <polarai/operation/internal/Conv2DOperationInternal.hpp>

using namespace polarai::core::internal;

namespace polarai::operation::internal {
Conv2DOperationInternal::Conv2DOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)) {}

utils::Index Conv2DOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors) const {
  // TODO support multiple channels
  // TODO support different paddings
  auto dataType = tensors[0]->getDataType();
  size_t x =
      tensors[0]->getShapeView().dim(0) - tensors[1]->getShapeView().dim(0) + 1;
  size_t y =
      tensors[0]->getShapeView().dim(1) - tensors[1]->getShapeView().dim(1) + 1;
  core::TensorShape tensorShape({x, y});
  // TODO check preconditions
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), dataType, std::move(tensorShape));
}

core::internal::GenValue Conv2DOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    core::internal::Generator& generator,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors,
    const core::internal::TensorInternal* resultTensor,
    core::internal::GenNode parentNode) const {
  generator.setInsertionPoint(parentNode);

  std::unordered_map<utils::Index, GenValue> argMap;
  GenValue a = parentNode.getOperand(
      mapMarkToLocalTensorIndex.at(Conv2DOperation::INPUT));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(Conv2DOperation::INPUT))
             ->getPublicIndex()] = a;
  GenValue b = parentNode.getOperand(
      mapMarkToLocalTensorIndex.at(Conv2DOperation::KERNEL));
  argMap[tensors.at(mapMarkToLocalTensorIndex.at(Conv2DOperation::KERNEL))
             ->getPublicIndex()] = b;

  std::unordered_map<utils::Index, GenValue> resultMap;
  GenValue out = parentNode.getResult();
  resultMap[resultTensor->getPublicIndex()] = out;

  lockTensors(generator, argMap, resultMap);

  generator.callBuiltin<builtin::Conv2D>(a, b, out);

  releaseTensors(generator, argMap, resultMap);

  return out;
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
Conv2DOperationInternal::genDerivative(
    const core::NodeState* inputNodeState,
    const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  return {};
}

size_t Conv2DOperationInternal::getOperandsCount() const { return 2; }
} // namespace polarai::operation::internal
