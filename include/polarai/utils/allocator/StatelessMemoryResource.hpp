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
#include <polarai/utils/allocator/AbstractMemoryResource.hpp>

namespace polarai::utils {
class POLAR_UTILS_EXPORT StatelessMemoryResource
    : public AbstractMemoryResource {
protected:
  byte* doAllocate(size_t size, size_t alignment) override;
  void doDeallocate(const byte* data, size_t size, size_t alignment) override;
};

} // namespace polarai::utils
