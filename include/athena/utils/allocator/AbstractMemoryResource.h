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

#ifndef ATHENA_ABSTRACTMEMORYRESOURCE_H
#define ATHENA_ABSTRACTMEMORYRESOURCE_H

#include <athena/utils/utils_export.h>
#include <cstddef>

namespace athena::utils {
using byte = void;

class ATH_UTILS_EXPORT AbstractMemoryResource {
public:
  virtual ~AbstractMemoryResource() = default;
  byte* allocate(size_t size, size_t alignment);
  void deallocate(const byte* data, size_t size, size_t alignment);

protected:
  virtual byte* doAllocate(size_t size, size_t alignment) = 0;
  virtual void doDeallocate(const byte* data, size_t size,
                            size_t alignment) = 0;
};
} // namespace athena::utils

#endif // ATHENA_ABSTRACTMEMORYRESOURCE_H
