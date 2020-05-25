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

#include <athena/core/tensor/ShapeView.h>
#include <athena/core/tensor/TensorShape.h>

namespace athena::core {
ShapeView::ShapeView(Iterator begin, Iterator end) : mBegin(begin), mEnd(end) {}
ShapeView::ShapeView(const TensorShape& shape)
    : mBegin(shape.begin()), mEnd(shape.end()) {}
size_t ShapeView::operator[](size_t index) const { return dim(index); }
bool ShapeView::operator==(const TensorShape& rhs) const {
  return *this == ShapeView(rhs);
}
bool ShapeView::operator==(const ShapeView& rhs) const {
  if (dimensions() != rhs.dimensions()) {
    return false;
  }
  for (size_t index = 0; index < dimensions(); ++index) {
    if (dim(index) != rhs.dim(index)) {
      return false;
    }
  }
  return true;
}
bool ShapeView::operator!=(const TensorShape& rhs) const {
  return !(*this == rhs);
}
bool ShapeView::operator!=(const ShapeView& rhs) const {
  return !(*this == rhs);
}
TensorShape ShapeView::toShape() const { return TensorShape(mBegin, mEnd); }
ShapeView::Iterator ShapeView::begin() const { return mBegin; }
ShapeView::Iterator ShapeView::end() const { return mEnd; }
size_t ShapeView::dim(size_t index) const { return *(mBegin + index); }
size_t ShapeView::dimensions() const {
  return static_cast<size_t>(std::distance(mBegin, mEnd));
}
size_t ShapeView::getTotalSize() const {
  size_t totalSize = 1;
  for (auto it = mBegin; it != mEnd; ++it) {
    totalSize *= *it;
  }
  return std::distance(mBegin, mEnd) == 0 ? 0 : totalSize;
}
ShapeView ShapeView::getSubShapeView(size_t offset) const {
  return ShapeView(mBegin + offset, mEnd);
}
} // namespace athena::core
