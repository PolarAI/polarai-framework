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

#include <polarai/utils/error/FatalError.hpp>

#include <cstddef>

namespace polarai::core {
enum class DataType : int { UNDEFINED = 0, DOUBLE = 1, FLOAT = 2, HALF = 3 };

inline size_t sizeOfDataType(const DataType& dataType) {
  switch (dataType) {
  case DataType::DOUBLE:
    return 8ULL;
  case DataType::FLOAT:
    return 4ULL;
  case DataType::HALF:
    return 2ULL;
  default:
    utils::FatalError(utils::ATH_FATAL_OTHER,
                      "Size for dataType is not defined");
    return 0;
  }
}
} // namespace polarai::core
