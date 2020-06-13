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

#ifndef ATHENA_TENSORINTERNAL_H
#define ATHENA_TENSORINTERNAL_H

#include <athena/core/Entity.h>
#include <athena/core/core_export.h>
#include <athena/core/tensor/DataType.h>
#include <athena/core/tensor/TensorShape.h>
#include <athena/utils/Index.h>
#include <athena/utils/Pointer.h>

namespace athena::core::internal {
class ContextInternal;

class ATH_CORE_EXPORT TensorInternal : public Entity {
public:
  TensorInternal(const TensorInternal& rhs) = default;
  TensorInternal(TensorInternal&& rhs) = default;
  explicit TensorInternal(utils::WeakPtr<ContextInternal> context,
                          utils::Index publicIndex, DataType dataType,
                          TensorShape shape);
  ~TensorInternal() override = default;

  // TensorInternal operator[](size_t index) const;

  [[nodiscard]] DataType getDataType() const;
  [[nodiscard]] ShapeView getShapeView() const;
  [[nodiscard]] ShapeView getSubShapeView(size_t offset = 1) const;
  [[nodiscard]] const TensorShape& getShape() const;
  [[nodiscard]] size_t getSize() const;
  void setShape(TensorShape shape);
  utils::Index getVirtualAddress() const;

private:
  DataType mDataType;
  TensorShape mShape;
  utils::Index mVirtualAddress;
};
} // namespace athena::core::internal

#endif // ATHENA_TENSORINTERNAL_H
