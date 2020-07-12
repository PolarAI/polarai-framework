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

#include <polarai/utils/allocator/AbstractMemoryResource.hpp>

namespace polarai::utils {
byte* AbstractMemoryResource::allocate(size_t size, size_t alignment) {
  return doAllocate(size, alignment);
}

void AbstractMemoryResource::deallocate(const byte* data, size_t size,
                                        size_t alignment) {
  doDeallocate(data, size, alignment);
}
} // namespace polarai::utils
