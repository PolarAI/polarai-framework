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

#ifndef ATHENA_TENSORSHAPE_H
#define ATHENA_TENSORSHAPE_H

#include "athena/utils/error/FatalError.h"

#include <athena/core/core_export.h>
#include <athena/core/tensor/ShapeView.h>

#include <initializer_list>
#include <utility>
#include <vector>

namespace athena::core {
class ATH_CORE_EXPORT TensorShape {
private:
  std::vector<size_t> mShape;
  size_t mTotalSize;
  size_t calculateTotalSize();

public:
  using Iterator = std::vector<size_t>::const_iterator;
  TensorShape();
  TensorShape(std::initializer_list<size_t> rhs);
  explicit TensorShape(std::vector<size_t> rhs);
  template <typename CopyableIterator>
  TensorShape(CopyableIterator begin, CopyableIterator end);
  TensorShape(const TensorShape& rhs) = default;
  TensorShape(TensorShape&& rhs) noexcept = default;
  ~TensorShape() = default;

  TensorShape& operator=(const TensorShape& rhs) = default;
  TensorShape& operator=(TensorShape&& rhs) = default;
  size_t operator[](size_t);
  bool operator==(const TensorShape& rhs) const;
  bool operator==(const ShapeView& rhs) const;
  bool operator!=(const TensorShape& rhs) const;
  bool operator!=(const ShapeView& rhs) const;

  /**
   * Returns shape as std::vector
   * @return vector containing sizes for every dimension
   */
  const std::vector<size_t>& getShape() const;
  ShapeView getShapeView() const;
  TensorShape getSubShape(size_t offset = 1) const;
  ShapeView getSubShapeView(size_t offset = 1) const;

  /**
   * Returns number of elements in Tensor
   * @return number of elements in Tensor
   */
  size_t getTotalSize() const;

  /**
   * Returns size for certain dimension
   * @param index Dimension index ( 0 <= index < dimensions )
   * @return Size for dimension
   */
  size_t dim(size_t index) const;

  /**
   * Returns number of dimensions in the shape
   * @return Number of dimensions
   */
  size_t dimensions() const;

  /**
   * Returns clone of mShape without first element
   * @return Clone of mShape without first element
   */
  Iterator begin() const;
  Iterator end() const;
};

template <typename CopyableIterator>
ATH_CORE_EXPORT TensorShape::TensorShape(CopyableIterator begin,
                                         CopyableIterator end)
    : mShape(begin, end), mTotalSize(calculateTotalSize()) {}

} // namespace athena::core

#endif // ATHENA_TENSORSHAPE_H
