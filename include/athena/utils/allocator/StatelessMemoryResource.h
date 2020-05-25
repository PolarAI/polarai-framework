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

#ifndef ATHENA_STATELESSMEMORYRESOURCE_H
#define ATHENA_STATELESSMEMORYRESOURCE_H

#include <athena/utils/allocator/AbstractMemoryResource.h>

namespace athena::utils {
class ATH_UTILS_EXPORT StatelessMemoryResource : public AbstractMemoryResource {
protected:
  byte* doAllocate(size_t size, size_t alignment) override;
  void doDeallocate(const byte* data, size_t size, size_t alignment) override;
};

} // namespace athena::utils

#endif // ATHENA_STATELESSMEMORYRESOURCE_H
