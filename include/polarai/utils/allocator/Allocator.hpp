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

#pragma once

#include <polar_utils_export.h>
#include <polarai/utils/Pointer.hpp>
#include <polarai/utils/allocator/AbstractMemoryResource.hpp>
#include <polarai/utils/allocator/StatelessMemoryResource.hpp>

#include <cstddef>
#include <memory>
#include <type_traits>

namespace polarai::utils {
template <typename T> class POLAR_UTILS_EXPORT Allocator {
public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::false_type;

  explicit Allocator(SharedPtr<AbstractMemoryResource> memoryResource =
                         makeShared<StatelessMemoryResource>())
      : mMemoryResource(std::move(memoryResource)) {}

  Allocator(const Allocator&) = default;

  Allocator(Allocator&&) = default;

  ~Allocator() = default;

  [[nodiscard]] T* allocate(size_t n) {
    return static_cast<T*>(allocate_bytes(n * sizeof(T), alignof(T)));
  }

  void deallocate(T* ptr, size_t n) {
    deallocate_bytes(ptr, n * sizeof(T), alignof(T));
  }

  template <typename U, typename... Args>
  void construct(U* ptr, Args&&... args) {
    new (ptr) U(std::forward<Args&&>(args)...);
  }

  template <typename U> void destroy(U* ptr) { ptr->~U(); }

  [[nodiscard]] void*
  allocate_bytes(size_t size, size_t alignment = alignof(std::max_align_t)) {
    return mMemoryResource->allocate(size, alignment);
  }

  void deallocate_bytes(byte* data, size_t size,
                        size_t alignment = alignof(std::max_align_t)) {
    mMemoryResource->deallocate(data, size, alignment);
  }

  SharedPtr<AbstractMemoryResource>& getMemoryResource() {
    return mMemoryResource;
  }

private:
  SharedPtr<AbstractMemoryResource> mMemoryResource;
};
} // namespace polarai::utils
