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

#include <polarai/core/tensor/internal/TensorInternal.hpp>
#include <polar_core_export.h>

#include <cstddef>

namespace polarai::core::internal {

enum class LockType { READ, READ_WRITE };

class POLAR_CORE_EXPORT TensorAllocator {
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

} // namespace polarai::core::internal
