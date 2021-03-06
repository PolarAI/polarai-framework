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
#include <polarai/core/graph/internal/GraphInternal.hpp>

#include <iostream>

namespace polarai::core::internal {
ContextInternal::ContextInternal(utils::Allocator<utils::byte> allocator,
                                 size_t defaultCapacity,
                                 size_t elementAverageSize)
    : mAllocator(std::move(allocator)), mContainer(),
      mPublicIndexToPrivateIndex(), mInstancesCounter{1},
      mNextTensorVirtualAddress{1} {
  mContainer.emplace_back(defaultCapacity, elementAverageSize, mAllocator);
}

ContextInternal::~ContextInternal() {}

const Traversal& ContextInternal::traverse(utils::Index publicGraphIndex) {
  auto& graph = getRef<GraphInternal>(publicGraphIndex);
  return graph.traverse();
}

utils::Allocator<utils::byte>& ContextInternal::getAllocator() {
  return mAllocator;
}

utils::Index ContextInternal::getNextPublicIndex() const {
  return mInstancesCounter;
}

utils::Index ContextInternal::registerTensor(const TensorInternal& tensor) {
  auto requiredSize = tensor.getSize() * sizeOfDataType(tensor.getDataType());
  auto returnedIndex = mNextTensorVirtualAddress;
  mNextTensorVirtualAddress += requiredSize;
  return returnedIndex;
}
} // namespace polarai::core::internal
