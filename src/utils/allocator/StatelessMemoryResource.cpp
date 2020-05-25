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

#include <athena/utils/allocator/StatelessMemoryResource.h>
#include <iostream>

namespace athena::utils {
byte* StatelessMemoryResource::doAllocate(size_t size, size_t alignment) {
  auto tmp = new unsigned char[size];
  // std::cout << "ALLOC: " << static_cast<void*>(tmp) << std::endl;
  return tmp;
}

void StatelessMemoryResource::doDeallocate(const byte* data, size_t size,
                                           size_t alignment) {
  // std::cout << "DEALLOC: " << data << std::endl;
  delete[] reinterpret_cast<const char*>(data);
}
} // namespace athena::utils
