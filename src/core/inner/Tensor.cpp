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

#include <athena/core/FatalError.h>
#include <athena/core/inner/GlobalTables.h>
#include <athena/core/inner/Tensor.h>

#include <string>

namespace athena::core::inner {
Tensor::Tensor(const Tensor& rhs) : mShapeOffset(0), mShapePartialProduct(1) {
    mVirtualAddress = getAllocationTable().registerRecord(
        rhs.getDataType(), rhs.getShapeView().toShape());
    mRecordIndex = getAllocationTable().size() - 1;
}
Tensor::Tensor(DataType dataType, TensorShape shape)
    : mVirtualAddress(
          getAllocationTable().registerRecord(dataType, std::move(shape))),
      mRecordIndex(getAllocationTable().size() - 1),
      mShapeOffset(0),
      mShapePartialProduct(1) {}
Tensor::Tensor(size_t id,
               size_t recordIndex,
               size_t shapeOffset,
               size_t shapePartialProduct)
    : mVirtualAddress(id),
      mRecordIndex(recordIndex),
      mShapeOffset(shapeOffset),
      mShapePartialProduct(shapePartialProduct) {}
Tensor& Tensor::operator=(const Tensor& rhs) {
    mShapeOffset = 0;
    mVirtualAddress = getAllocationTable().registerRecord(
        rhs.getDataType(), rhs.getShapeView().toShape());
    mRecordIndex = inner::getAllocationTable().size() - 1;
    mShapePartialProduct = 1;
    return *this;
}
Tensor Tensor::operator[](size_t index) const {
    auto shapeView = getAllocationTable()[mRecordIndex].getShapeView();
    size_t subShapePartialProduct =
        shapeView.dim(mShapeOffset) * mShapePartialProduct;
    return Tensor(mVirtualAddress +
                      shapeView.getTotalSize() / subShapePartialProduct * index,
                  mRecordIndex, mShapeOffset + 1, subShapePartialProduct);
}
DataType Tensor::getDataType() const {
    return getAllocationTable()[mRecordIndex].getDataType();
}
ShapeView Tensor::getShapeView() const {
    return getAllocationTable()[mRecordIndex].getShapeView(mShapeOffset);
}
ShapeView Tensor::getSubShapeView(size_t offset) const {
    return getAllocationTable()[mRecordIndex].getShapeView(mShapeOffset +
                                                           offset);
}
size_t Tensor::getVirtualAddress() const {
    return mVirtualAddress;
}
size_t Tensor::getSize() const {
    return getAllocationTable()[mRecordIndex].getSize() / mShapePartialProduct;
}
void Tensor::setShape(const TensorShape& shape) {
    getAllocationTable()[mRecordIndex] = AllocationRecord(
        getAllocationTable()[mRecordIndex].getDataType(), shape);
}
void Tensor::clear() {
    mVirtualAddress = 0;
    mRecordIndex = inner::kKUndefinedIndex;
    mShapeOffset = 0;
    mShapePartialProduct = 1;
}
}  // namespace athena::core::inner
