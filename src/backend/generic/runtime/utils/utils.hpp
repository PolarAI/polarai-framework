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

#include <polarai/backend/generic/BackendAllocator.hpp>
#include <polarai/backend/generic/runtime/TensorInfo.h>

inline auto tensorInfoToRecord(TensorInfo* tensor)
    -> polarai::backend::generic::MemoryRecord {
  polarai::backend::generic::MemoryRecord record;
  record.virtualAddress = tensor->virtAddr;
  record.allocationSize = polarai::core::sizeOfDataType(
      static_cast<polarai::core::DataType>(tensor->dataType));
  for (int i = 0; i < tensor->dims; i++) {
    record.allocationSize *= tensor->shape[i];
  }
  return record;
}

inline auto tensorSize(TensorInfo* tensor) {
  uint64_t total = 1;
  for (int i = 0; i < tensor->dims; i++) {
    total *= tensor->shape[i];
  }
  return total;
}
