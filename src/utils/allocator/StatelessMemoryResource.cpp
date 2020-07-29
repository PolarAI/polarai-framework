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
#include <polarai/utils/allocator/StatelessMemoryResource.hpp>

namespace polarai::utils {
void* StatelessMemoryResource::doAllocate(size_t size, size_t alignment) {
  return utils::allocate(size, alignment);
}

void StatelessMemoryResource::doDeallocate(void* data, size_t size,
                                           size_t alignment) {
  utils::deallocate(data, size, alignment);
}
} // namespace polarai::utils
