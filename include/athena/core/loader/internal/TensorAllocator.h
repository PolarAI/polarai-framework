//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#ifndef ATHENA_TENSOR_ALLOCATOR_H
#define ATHENA_TENSOR_ALLOCATOR_H

#include <athena/core/core_export.h>
#include <athena/core/tensor/internal/TensorInternal.h>

#include <cstddef>

namespace athena::core::internal {

enum class LockType { READ, READ_WRITE };

class ATH_CORE_EXPORT TensorAllocator {
public:
  virtual ~TensorAllocator() = default;

  /// Allocates memory for Tensor.
  virtual void allocate(const TensorInternal& tensor) = 0;

  /// Returns memory to system.
  virtual void deallocate(const TensorInternal& tensor) = 0;

  /// \return a pointer to raw Tensor data.
  virtual void* get(const TensorInternal& tensor) = 0;

  /// Locks tensor in RAM.
  ///
  /// Locked tensors can not be moved to other memory domain or deallocated.
  virtual void lock(const TensorInternal& tensor, LockType type) = 0;

  /// Releases tensor memory object.
  virtual void release(const TensorInternal& tensor) = 0;
};

} // namespace athena::core::internal

#endif // ATHENA_TENSOR_ALLOCATOR_H
