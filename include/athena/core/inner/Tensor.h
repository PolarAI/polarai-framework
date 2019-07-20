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

#ifndef ATHENA_TENSOR_H
#define ATHENA_TENSOR_H

#include <athena/core/inner/AllocationRecord.h>

namespace athena::core::inner {
class Tensor {
    private:
    size_t mVirtualAddress;
    size_t mRecordIndex;
    size_t mShapeOffset;
    size_t mShapePartialProduct;
    Tensor(size_t id,
           size_t recordIndex,
           size_t shapeOffset,
           size_t shapePartialProduct);

    public:
    Tensor(const Tensor& rhs) = default;
    Tensor(Tensor&& rhs) noexcept = default;
    Tensor(DataType dataType, TensorShape shape);
    ~Tensor() = default;

    Tensor& operator=(const Tensor& rhs) = default;
    Tensor& operator=(Tensor&& rhs) noexcept = default;
    Tensor operator[](size_t index) const;

    DataType getDataType() const;
    ShapeView getShapeView() const;
    ShapeView getSubShapeView(size_t offset = 1) const;
    const TensorShape& getShape() const;
    size_t getVirtualAddress() const;
    size_t getSize() const;
    void setShape(const TensorShape& shape);
    void clear();
};
}  // namespace athena::core::inner

#endif  // ATHENA_TENSOR_H
