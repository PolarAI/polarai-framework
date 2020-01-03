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

#include <athena/core/inner/AllocationRecord.h>

#include <numeric>

namespace athena::core::inner {
AllocationRecord::AllocationRecord(DataType dataType, TensorShape shape)
    : mDataType(dataType), mShape(std::move(shape)) {}
DataType AllocationRecord::getDataType() const { return mDataType; }
const TensorShape& AllocationRecord::getShape() const { return mShape; }
ShapeView AllocationRecord::getShapeView(size_t offset) const {
  return mShape.getSubShapeView(offset);
}
size_t AllocationRecord::getSize() const { return mShape.getTotalSize(); }
} // namespace athena::core::inner
