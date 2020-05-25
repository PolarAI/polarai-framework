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

#include <athena/core/tensor/TensorShape.h>
#include <athena/utils/error/FatalError.h>

#include <numeric>
#include <stdexcept>
#include <string>

namespace athena::core {
TensorShape::TensorShape() : mShape{}, mTotalSize(0) {}
TensorShape::TensorShape(std::initializer_list<size_t> rhs)
    : mShape(rhs), mTotalSize(calculateTotalSize()) {}
TensorShape::TensorShape(std::vector<size_t> rhs)
    : mShape(std::move(rhs)), mTotalSize(calculateTotalSize()) {}
size_t TensorShape::operator[](size_t index) { return dim(index); }
bool TensorShape::operator==(const TensorShape& rhs) const {
  return ShapeView(*this) == ShapeView(rhs);
}
bool TensorShape::operator==(const ShapeView& rhs) const {
  return ShapeView(*this) == rhs;
}
bool TensorShape::operator!=(const TensorShape& rhs) const {
  return !(*this == rhs);
}
bool TensorShape::operator!=(const ShapeView& rhs) const {
  return !(*this == rhs);
}
size_t TensorShape::calculateTotalSize() {
  size_t totalSize = std::accumulate(mShape.begin(), mShape.end(), 1,
// todo remove hacks when gcc is updated
#ifdef __clang__
                                     std::multiplies());
#else
                                     std::multiplies<int>());
#endif
  return mShape.empty() ? 0 : totalSize;
}
const std::vector<size_t>& TensorShape::getShape() const { return mShape; }
ShapeView TensorShape::getShapeView() const {
  return ShapeView(mShape.begin(), mShape.end());
}
TensorShape TensorShape::getSubShape(size_t offset) const {
  return TensorShape(mShape.begin() + offset, mShape.end());
}
ShapeView TensorShape::getSubShapeView(size_t offset) const {
  return ShapeView(mShape.begin() + offset, mShape.end());
}
size_t TensorShape::getTotalSize() const { return mTotalSize; }
size_t TensorShape::dim(size_t index) const { return mShape[index]; }
size_t TensorShape::dimensions() const { return mShape.size(); }
TensorShape::Iterator TensorShape::begin() const { return mShape.begin(); }
TensorShape::Iterator TensorShape::end() const { return mShape.end(); }
} // namespace athena::core
