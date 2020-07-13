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

#include <polarai/core/context/internal/ContextInternal.hpp>
#include <polarai/core/tensor/internal/TensorInternal.hpp>

#include <iostream>

namespace polarai::core::internal {
TensorInternal::TensorInternal(utils::WeakPtr<ContextInternal> context,
                               utils::Index publicIndex, DataType dataType,
                               TensorShape shape)
    : Entity(std::move(context), publicIndex), mDataType(dataType),
      mShape(std::move(shape)),
      mVirtualAddress(mContext.lock()->registerTensor(*this)) {}

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

} // namespace polarai::core::internal
