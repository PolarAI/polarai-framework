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

#include <cstddef>
#include <functional>

namespace polarai::backend::generic {
struct MemoryRecord {
  size_t virtualAddress;
  size_t allocationSize;
  bool operator==(const MemoryRecord& record) const {
    return virtualAddress == record.virtualAddress;
  }
};
} // namespace polarai::backend::generic

namespace std {
template <> class hash<polarai::backend::generic::MemoryRecord> {
public:
  size_t
  operator()(const polarai::backend::generic::MemoryRecord& record) const {
    return record.virtualAddress;
  }
};
} // namespace std
