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

#ifndef ATHENA_ALLOCATOR_H
#define ATHENA_ALLOCATOR_H

#include <athena/core/core_export.h>
#include <athena/core/inner/Tensor.h>

#include <cstddef>

namespace athena::core {

/// Basic Allocator interface for backends.
class ATH_CORE_EXPORT Allocator {
public:
  virtual ~Allocator() = default;

  /// Allocates memory for Tensor.
  virtual void allocate(const inner::Tensor& tensor) = 0;

  /// Returns memory to system.
  virtual void deallocate(const inner::Tensor& tensor) = 0;

  /// \return a pointer to raw Tensor data.
  virtual void* get(const inner::Tensor& tensor) = 0;

  /// Locks tensor in RAM.
  ///
  /// Locked tensors can not be moved to other memory domain or deallocated.
  virtual void lock(const inner::Tensor& tensor) = 0;

  /// Releases tensor memory object.
  virtual void release(const inner::Tensor& tensor) = 0;
};

} // namespace athena::core

#endif // ATHENA_ALLOCATOR_H
