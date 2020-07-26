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
#include <polarai/loaders/ConstantLoader.hpp>
#include <polarai/loaders/DummyLoader.hpp>
#include <polarai/operation/CombineOperation.hpp>
#include <polarai/operation/MulOperation.hpp>
#include <polarai/operation/Pool2DOperation.hpp>

using namespace polarai::core::internal;

namespace polarai::operation::internal {
Pool2DOperationInternal::Pool2DOperationInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, std::initializer_list<int64_t> sizes,
    std::initializer_list<int64_t> stride, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)),
      mSizes(sizes), mStrides(stride) {}

utils::Index Pool2DOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    const std::unordered_map<int64_t, utils::Index>& mapMarkToLocalTensorIndex,
    const std::vector<core::internal::TensorInternal*>& tensors) const {
  auto dataType = tensors[0]->getDataType();
  size_t dim0 =
      (tensors[0]->getShapeView().dim(0) - mSizes[0]) / mStrides[0] + 1;
  size_t dim1 =
      (tensors[0]->getShapeView().dim(1) - mSizes[1]) / mStrides[1] + 1;
  core::TensorShape tensorShape{dim0, dim1};
  // TODO check preconditions
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), dataType, std::move(tensorShape));
}

core::internal::GenValue Pool2DOperationInternal::gen(
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

  generator.setInsertionPoint(parentNode);

  lockTensors(generator, argMap, resultMap);

  generator.callBuiltin<builtin::Pool2D>(input, out, mSizes, mStrides);

  releaseTensors(generator, argMap, resultMap);

  return out;
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
Pool2DOperationInternal::genDerivative(
    const core::NodeState* inputNodeState,
    const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  return {};
}

size_t Pool2DOperationInternal::getOperandsCount() const { return 2; }
} // namespace polarai::operation::internal
