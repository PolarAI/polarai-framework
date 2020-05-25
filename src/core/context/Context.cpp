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

#include <athena/core/context/Context.h>
#include <athena/core/context/internal/ContextInternal.h>

// void* operator new(std::size_t sz) {
//  void *ptr = std::malloc(sz);
//  //std::cout << "ALLOC: " << static_cast<void*>(ptr) << std::endl;
//  return ptr;
//}
// void operator delete(void* ptr) noexcept
//{
//  //std::cout << "DEALLOC: " << static_cast<void*>(ptr) << std::endl;
//  std::free(ptr);
//}

namespace athena::core {
Context::Context(utils::Allocator allocator, size_t defaultCapacity,
                 size_t elementAverageSize)
    : mInternal(utils::makeShared<internal::ContextInternal>(
          std::move(allocator), defaultCapacity, elementAverageSize)) {}

Context::Context(utils::SharedPtr<internal::ContextInternal> ptr)
    : mInternal(std::move(ptr)) {}

Context::~Context() {}

utils::SharedPtr<internal::ContextInternal> Context::internal() {
  return mInternal;
}

utils::SharedPtr<const internal::ContextInternal> Context::internal() const {
  return mInternal;
}

utils::Allocator& Context::getAllocator() { return mInternal->getAllocator(); }
} // namespace athena::core