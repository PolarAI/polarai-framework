//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
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

namespace athena::backend::llvm {
struct MemoryRecord {
  size_t virtualAddress;
  size_t allocationSize;
  bool operator==(const MemoryRecord& record) const {
    return virtualAddress == record.virtualAddress;
  }
};
} // namespace athena::backend::llvm

namespace std {
template <> class hash<athena::backend::llvm::MemoryRecord> {
public:
  size_t operator()(const athena::backend::llvm::MemoryRecord& record) const {
    return record.virtualAddress;
  }
};
} // namespace std
