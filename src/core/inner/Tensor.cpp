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
#include <athena/core/inner/Tensor.h>
#include <athena/core/inner/InnerFunctions.h>
#include <athena/core/inner/Table.h>

#include <string>

namespace athena::core::inner {
Tensor::Tensor(DataType dataType, TensorShape shape, Context& context)
    : mContext(&context),
      mVirtualAddress(
          getAllocationTable(context).registerRecord(dataType, std::move(shape))),
      mRecordIndex(getAllocationTable(context).size() - 1),
      mShapeOffset(0),
      mShapePartialProduct(1) {}
Tensor::Tensor(Context& context,
               size_t id,
               size_t recordIndex,
               size_t shapeOffset,
               size_t shapePartialProduct)
    : mContext(&context),
      mVirtualAddress(id),
      mRecordIndex(recordIndex),
      mShapeOffset(shapeOffset),
      mShapePartialProduct(shapePartialProduct) {}
Tensor Tensor::operator[](size_t index) const {
    auto shapeView = getAllocationTable(*mContext)[mRecordIndex].getShapeView();
    size_t subShapePartialProduct =
        shapeView.dim(mShapeOffset) * mShapePartialProduct;
    return Tensor(*mContext, mVirtualAddress +
                      shapeView.getTotalSize() / subShapePartialProduct * index,
                  mRecordIndex, mShapeOffset + 1, subShapePartialProduct);
}
DataType Tensor::getDataType() const {
    return getAllocationTable(*mContext)[mRecordIndex].getDataType();
}
ShapeView Tensor::getShapeView() const {
    return getAllocationTable(*mContext)[mRecordIndex].getShapeView(mShapeOffset);
}
ShapeView Tensor::getSubShapeView(size_t offset) const {
    return getAllocationTable(*mContext)[mRecordIndex].getShapeView(mShapeOffset +
                                                           offset);
}
const TensorShape& Tensor::getShape() const {
    if (mShapeOffset != 0) {
        FatalError(ATH_NOT_IMPLEMENTED,
                   "getShape is not supported for subtensors");
    }
    return getAllocationTable(*mContext)[mRecordIndex].getShape();
}
size_t Tensor::getVirtualAddress() const {
    return mVirtualAddress;
}
size_t Tensor::getSize() const {
    return getAllocationTable(*mContext)[mRecordIndex].getSize() / mShapePartialProduct;
}
void Tensor::setShape(const TensorShape& shape) {
    getAllocationTable(*mContext)[mRecordIndex] = AllocationRecord(
        getAllocationTable(*mContext)[mRecordIndex].getDataType(), shape);
}
void Tensor::clear() {
    mVirtualAddress = 0;
    mRecordIndex = inner::kKUndefinedIndex;
    mShapeOffset = 0;
    mShapePartialProduct = 1;
}

Tensor* getNullTensor(Context& context) {
    return new Tensor(DataType::UNDEFINED, {0}, context);
}

}  // namespace athena::core::inner
