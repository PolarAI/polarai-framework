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
#include <polarai/utils/Memory.hpp>

#include <cstddef>

namespace polarai::utils {
class POLAR_UTILS_EXPORT AbstractMemoryResource {
public:
  virtual ~AbstractMemoryResource() = default;
  byte* allocate(size_t size, size_t alignment);
  void deallocate(byte* data, size_t size, size_t alignment);

protected:
  virtual void* doAllocate(size_t size, size_t alignment) = 0;
  virtual void doDeallocate(void* data, size_t size, size_t alignment) = 0;
};
} // namespace polarai::utils
