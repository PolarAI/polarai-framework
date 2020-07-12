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

#include <polarai/utils/allocator/StatelessMemoryResource.hpp>

#include <iostream>

namespace polarai::utils {
byte* StatelessMemoryResource::doAllocate(size_t size, size_t alignment) {
  auto tmp = new unsigned char[size];
  return tmp;
}

void StatelessMemoryResource::doDeallocate(const byte* data, size_t size,
                                           size_t alignment) {
  delete[] reinterpret_cast<const char*>(data);
}
} // namespace polarai::utils
