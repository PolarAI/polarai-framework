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

#include <athena/core/context/internal/ContextInternal.h>
#include <athena/core/tensor/internal/TensorInternal.h>
#include <iostream>

namespace athena::core::internal {
TensorInternal::TensorInternal(utils::WeakPtr<ContextInternal> context,
                               utils::Index publicIndex, DataType dataType,
                               TensorShape shape)
    : Entity(std::move(context), publicIndex), mDataType(dataType),
      mShape(std::move(shape)),
      mVirtualAddress(mContext.lock()->registerTensor(*this)) {}

// TensorInternal::TensorInternal(DataType dataType, TensorShape shape, Context&
// context)
//    : mContext(&context),
//      mVirtualAddress(getAllocationTable(context).registerRecord(
//          dataType, std::move(shape))),
//      mRecordIndex(getAllocationTable(context).size() - 1), mShapeOffset(0),
//      mShapePartialProduct(1) {}

// TensorInternal::TensorInternal(Context& context, size_t id, size_t
// recordIndex,
//               size_t shapeOffset, size_t shapePartialProduct)
//    : mContext(&context), mVirtualAddress(id), mRecordIndex(recordIndex),
//      mShapeOffset(shapeOffset), mShapePartialProduct(shapePartialProduct) {}

// TensorInternal TensorInternal::operator[](size_t index) const {
//  auto shapeView = getAllocationTable(*mContext)[mRecordIndex].getShapeView();
//  size_t subShapePartialProduct =
//      shapeView.dim(mShapeOffset) * mShapePartialProduct;
//  return Tensor(*mContext,
//                mVirtualAddress +
//                    shapeView.getTotalSize() / subShapePartialProduct * index,
//                mRecordIndex, mShapeOffset + 1, subShapePartialProduct);
//}

DataType TensorInternal::getDataType() const { return mDataType; }

ShapeView TensorInternal::getShapeView() const { return ShapeView(mShape); }

ShapeView TensorInternal::getSubShapeView(size_t offset) const {
#ifdef DEBUG
  if (mShape.begin() + offset > mShape.end()) {
    new utils::FatalError(utils::ATH_BAD_ACCESS,
                          "Bad SubShapeView constructing");
  }
#endif
  return ShapeView(mShape.begin() + offset, mShape.end());
}

const TensorShape& TensorInternal::getShape() const { return mShape; }

size_t TensorInternal::getSize() const { return mShape.getTotalSize(); }

void TensorInternal::setShape(TensorShape shape) {
  mShape = std::move(shape);
  mVirtualAddress = mContext.lock()->registerTensor(*this);
}

utils::Index TensorInternal::getVirtualAddress() const {
  return mVirtualAddress;
}

} // namespace athena::core::internal
