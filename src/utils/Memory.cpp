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

#include <polarai/utils/Memory.hpp>

#include <cstdlib>

#ifdef HAS_MIMALLOC
#include <mimalloc.h>
#endif

namespace polarai::utils {
void* bare_allocate(size_t size, size_t alignment, size_t offset) {
#ifdef HAS_MIMALLOC
  if (alignment % sizeof(void*) != 0) {
    alignment = alignof(std::max_align_t);
  }
  return mi_malloc_aligned_at(size, alignment, offset);
#else
  return aligned_alloc(alignment, size);
#endif
}
POLAR_UTILS_EXPORT void bare_deallocate(void* ptr, size_t size,
                                        size_t alignment) {
#ifdef HAS_MIMALLOC
  (void)size;
  return mi_free_aligned(ptr, alignment);
#else
  return std::free(ptr);
#endif
}
POLAR_UTILS_EXPORT void* allocate(size_t size, size_t alignment,
                                  size_t offset) {
  return bare_allocate(size, alignment, offset);
}
POLAR_UTILS_EXPORT void deallocate(void* ptr, size_t size, size_t alignment) {
  bare_deallocate(ptr, size, alignment);
}
} // namespace polarai::utils
