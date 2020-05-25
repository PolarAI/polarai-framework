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

#ifndef ATHENA_BACKENDALLOCATOR_H
#define ATHENA_BACKENDALLOCATOR_H

#include <athena/backend/llvm/runtime/Device.h>
#include <athena/core/loader/internal/TensorAllocator.h>
#include <athena/core/tensor/internal/TensorInternal.h>

using namespace athena::core::internal;

namespace athena::backend::llvm {
class BackendAllocator : public TensorAllocator {
protected:
  virtual void* getImpl(const TensorInternal& tensor, Device& device) = 0;

public:
  virtual void registerDevice(Device& device) = 0;

  /// Allocates memory on a particular device.
  virtual void allocate(const TensorInternal& tensor, Device& device) = 0;
  // fixme implement
  virtual void allocate(const MemoryRecord& record, Device& device){};

  /// Locks tensor raw memory on a particular device.
  virtual void lock(const TensorInternal& tensor, Device& device,
                    LockType type) = 0;
  // fixme implement
  virtual void lock(const MemoryRecord& record, Device& device,
                    LockType type){};
  virtual void release(const MemoryRecord& record, Device& device){};

  template <typename BufferT>
  BufferT* get(const TensorInternal& tensor, Device& device) {
    return reinterpret_cast<BufferT*>(getImpl(tensor, device));
  }
};
} // namespace athena::backend::llvm

#endif // ATHENA_BACKENDALLOCATOR_H
