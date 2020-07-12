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

#include <polarai/backend/generic/runtime/Device.hpp>
#include <polarai/core/loader/internal/TensorAllocator.hpp>
#include <polarai/core/tensor/internal/TensorInternal.hpp>

using namespace polarai::core::internal;

namespace polarai::backend::generic {
class BackendAllocator : public TensorAllocator {
protected:
  virtual void* getImpl(const TensorInternal& tensor, Device& device) = 0;
  virtual void* getImpl(const MemoryRecord& record, Device& device) = 0;

public:
  virtual void registerDevice(Device& device) = 0;

  /// Allocates memory on a particular device.
  virtual void allocate(const TensorInternal& tensor, Device& device) = 0;
  // fixme implement
  virtual void allocate(const MemoryRecord& record, Device& device) = 0;
  virtual void allocate(const MemoryRecord& record) = 0;

  /// Locks tensor raw memory on a particular device.
  virtual void lock(const TensorInternal& tensor, Device& device,
                    LockType type) = 0;
  virtual void lock(const MemoryRecord& record, Device& device,
                    LockType type) = 0;
  // For test purposes only
  virtual void lock(const MemoryRecord& record, LockType type) = 0;

  virtual void release(const MemoryRecord& record, Device& device) = 0;
  virtual void release(const MemoryRecord& record) = 0;

  template <typename BufferT>
  BufferT* get(const TensorInternal& tensor, Device& device) {
    return reinterpret_cast<BufferT*>(getImpl(tensor, device));
  }
  template <typename BufferT>
  BufferT* get(const MemoryRecord& record, Device& device) {
    return reinterpret_cast<BufferT*>(getImpl(record, device));
  }
  virtual void* get(const MemoryRecord& record) = 0;
};
} // namespace athena::backend::llvm
