/*
 * Copyright (c) 2019 Athena. All rights reserved.
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

#ifndef ATHENA_ALLOCATOR_H
#define ATHENA_ALLOCATOR_H

#include <athena/utils/Pointer.h>
#include <athena/utils/allocator/AbstractMemoryResource.h>
#include <athena/utils/allocator/StatelessMemoryResource.h>
#include <athena/utils/utils_export.h>

#include <cstddef>
#include <memory>

namespace athena::utils {
class ATH_UTILS_EXPORT Allocator {
public:
  //  template <typename MemoryResourceType, typename ...Args>
  //  explicit Allocator(Args&&... args);

  explicit Allocator(SharedPtr<AbstractMemoryResource> memoryResource =
                         makeShared<StatelessMemoryResource>());

  Allocator(const Allocator&) = default;

  Allocator(Allocator&&) = default;

  ~Allocator() = default;

  byte* allocateBytes(size_t size, size_t alignment = 64);

  void deallocateBytes(const byte* data, size_t size, size_t alignment = 64);

  SharedPtr<AbstractMemoryResource>& getMemoryResource();

private:
  SharedPtr<AbstractMemoryResource> mMemoryResource;
};

// template <typename MemoryResourceType, typename ...Args>
// Allocator::Allocator(Args&&... args)
//  :
//  mMemoryResource(makeShared<MemoryResourceType>(std::forward<Args>(args)...))
//  {
//}
} // namespace athena::utils

#endif // ATHENA_ALLOCATOR_H
