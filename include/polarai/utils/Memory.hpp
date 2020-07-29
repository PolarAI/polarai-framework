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

#include <cstddef>

namespace polarai::utils {
/// Allocate memory without any checks.
///
/// \param size is a size of allocation in bytes.
/// \param alignment is memory alignment in bytes.
POLAR_UTILS_EXPORT void*
bare_allocate(size_t size, size_t alignment = alignof(std::max_align_t),
              size_t offset = 0);
/// Deallocate memory without any checks.
///
/// \param ptr is a pointer to be freed. ptr must be allocated with
///        bare_allocate.
/// \param size is a size of allocation in bytes.
/// \param alignment is memory alignment in bytes.
POLAR_UTILS_EXPORT void
bare_deallocate(void* ptr, size_t size,
                size_t alignment = alignof(std::max_align_t));
/// Allocate memory with default allocator.
///
/// \param size is a size of allocation in bytes.
/// \param alignment is memory alignment in bytes.
POLAR_UTILS_EXPORT void* allocate(size_t size,
                                  size_t alignment = alignof(std::max_align_t),
                                  size_t offset = 0);
/// Deallocate memory with defaul allocator.
///
/// \param ptr is a pointer to be freed. ptr must be allocated with allocate.
/// \param size is a size of allocation in bytes.
/// \param alignment is memory alignment in bytes.
POLAR_UTILS_EXPORT void
deallocate(void* ptr, size_t size,
           size_t alignment = alignof(std::max_align_t));
} // namespace polarai::utils
