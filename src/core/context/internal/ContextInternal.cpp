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

#include <athena/core/context/internal/ContextInternal.h>
#include <athena/core/graph/internal/GraphInternal.h>
#include <iostream>

namespace athena::core::internal {
ContextInternal::ContextInternal(utils::Allocator allocator,
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

utils::Allocator& ContextInternal::getAllocator() { return mAllocator; }

utils::Index ContextInternal::getNextPublicIndex() const {
  return mInstancesCounter;
}

utils::Index ContextInternal::registerTensor(const TensorInternal& tensor) {
  auto requiredSize = tensor.getSize() * sizeOfDataType(tensor.getDataType());
  auto returnedIndex = mNextTensorVirtualAddress;
  mNextTensorVirtualAddress += requiredSize;
  return returnedIndex;
}
} // namespace athena::core::internal
