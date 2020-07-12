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

#include <polarai/utils/allocator/Allocator.hpp>

#include <utility>

namespace polarai::utils {
Allocator::Allocator(SharedPtr<AbstractMemoryResource> memoryResource)
    : mMemoryResource(std::move(memoryResource)) {}

byte* Allocator::allocateBytes(size_t size, size_t alignment) {
  return mMemoryResource->allocate(size, alignment);
}

void Allocator::deallocateBytes(const byte* data, size_t size,
                                size_t alignment) {
  mMemoryResource->deallocate(data, size, alignment);
}

SharedPtr<AbstractMemoryResource>& Allocator::getMemoryResource() {
  return mMemoryResource;
}
} // namespace polarai::utils
