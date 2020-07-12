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

namespace polarai::utils {
class POLAR_UTILS_EXPORT Allocator {
public:
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

} // namespace polarai::utils
