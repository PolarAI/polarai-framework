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

#ifndef ATHENA_ALLOCATIONTABLERECORD_H
#define ATHENA_ALLOCATIONTABLERECORD_H

#include <athena/core/DataType.h>
#include <athena/core/TensorShape.h>
#include <athena/core/core_export.h>

namespace athena::core::inner {
struct ATH_CORE_EXPORT AllocationRecord {
private:
  DataType mDataType;
  TensorShape mShape;

public:
  AllocationRecord(DataType dataType, TensorShape shape);
  DataType getDataType() const;
  const TensorShape& getShape() const;
  ShapeView getShapeView(size_t offset = 0) const;
  size_t getSize() const;
};
} // namespace athena::core::inner

#endif // ATHENA_ALLOCATIONTABLERECORD_H
