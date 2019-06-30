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

#ifndef ATHENA_SHAPEVIEW_H
#define ATHENA_SHAPEVIEW_H

#include <vector>

namespace athena::core {

class TensorShape;

class ShapeView {
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
}  // namespace athena::core

#endif  // ATHENA_SHAPEVIEW_H
