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
#include <polarai/core/Entity.hpp>
#include <polarai/core/tensor/DataType.hpp>
#include <polarai/core/tensor/TensorShape.hpp>
#include <polarai/utils/Index.hpp>
#include <polarai/utils/Pointer.hpp>

namespace polarai::core::internal {
class POLAR_CORE_EXPORT ContextInternal;

class POLAR_CORE_EXPORT TensorInternal : public Entity {
public:
  TensorInternal(const TensorInternal& rhs) = default;
  TensorInternal(TensorInternal&& rhs) = default;
  explicit TensorInternal(utils::WeakPtr<ContextInternal> context,
                          utils::Index publicIndex, DataType dataType,
                          TensorShape shape);
  ~TensorInternal() override = default;

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
} // namespace polarai::core::internal
