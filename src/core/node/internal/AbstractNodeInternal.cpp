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

#include <polarai/core/context/internal/ContextInternal.hpp>
#include <polarai/core/node/internal/AbstractNodeInternal.hpp>

namespace polarai::core::internal {
AbstractNodeInternal::AbstractNodeInternal(
    utils::WeakPtr<ContextInternal> context, utils::Index publicNodeIndex,
    utils::String name)
    : Entity(std::move(context), publicNodeIndex, std::move(name)),
      mTensorIndex{} {}

AbstractNodeInternal::AbstractNodeInternal(
    utils::WeakPtr<ContextInternal> context, utils::Index publicNodeIndex,
    utils::Index tensorIndex, utils::String name)
    : Entity(std::move(context), publicNodeIndex, std::move(name)),
      mTensorIndex(tensorIndex) {}

AbstractNodeInternal::~AbstractNodeInternal() {
  // TODO
}

void AbstractNodeInternal::after(const AbstractNodeInternal& node,
                                 EdgeMark mark) const {}

void AbstractNodeInternal::before(const AbstractNodeInternal& node,
                                  EdgeMark mark) const {}

void AbstractNodeInternal::clear() {
  // TODO
}

utils::Allocator AbstractNodeInternal::getAllocator() {
  return mContext.lock()->getAllocator();
}

const TensorInternal* AbstractNodeInternal::getTensorPtr() const {
  return mContext.lock()->getPtr<TensorInternal>(mTensorIndex);
}

TensorInternal* AbstractNodeInternal::getTensorPtr() {
  return mContext.lock()->getPtr<TensorInternal>(mTensorIndex);
}

utils::Index AbstractNodeInternal::getTensorIndex() const {
  return mTensorIndex;
}

void AbstractNodeInternal::setTensorIndex(utils::Index tensorIndex) {
  mTensorIndex = tensorIndex;
}
} // namespace polarai::core::internal
