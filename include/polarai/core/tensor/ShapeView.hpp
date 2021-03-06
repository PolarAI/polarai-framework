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

#include <polar_core_export.h>

#include <vector>

namespace polarai::core {

class TensorShape;

class POLAR_CORE_EXPORT ShapeView {
private:
  using Iterator = std::vector<size_t>::const_iterator;
  Iterator mBegin;
  Iterator mEnd;

public:
  ShapeView() = delete;
  ShapeView(const ShapeView& shapeView) = default;
  ShapeView(ShapeView&& shapeView) = default;
  ShapeView(Iterator begin, Iterator end);
  explicit ShapeView(const TensorShape& shape);
  ~ShapeView() = default;

  ShapeView& operator=(const ShapeView& shapeView) = default;
  ShapeView& operator=(ShapeView&& shapeView) = default;
  size_t operator[](size_t index) const;
  bool operator==(const TensorShape& rhs) const;
  bool operator==(const ShapeView& rhs) const;
  bool operator!=(const TensorShape& rhs) const;
  bool operator!=(const ShapeView& rhs) const;

  TensorShape toShape() const;
  Iterator begin() const;
  Iterator end() const;
  size_t dim(size_t index) const;
  size_t dimensions() const;
  size_t getTotalSize() const;
  ShapeView getSubShapeView(size_t offset = 1) const;
};
} // namespace polarai::core
