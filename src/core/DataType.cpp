/*
 * Copyright (c) 2018 Athena. All rights reserved.
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

#include <athena/core/DataType.h>
#include <athena/core/FatalError.h>

namespace athena::core {
size_t sizeOfDataType(const DataType& dataType) {
  switch (dataType) {
  case DataType::DOUBLE:
    return 8ULL;
  case DataType::FLOAT:
    return 4ULL;
  case DataType::HALF:
    return 2ULL;
  default:
    FatalError(ATH_FATAL_OTHER, "Size for dataType is not defined");
    return 0;
  }
}
} // namespace athena::core