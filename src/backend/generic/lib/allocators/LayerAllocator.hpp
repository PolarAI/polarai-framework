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

#include "TrivialAllocator.hpp"

#include <polarai/backend/generic/AllocatorLayerBase.hpp>
#include <polarai/backend/generic/BackendAllocator.hpp>
#include <polarai/backend/generic/MemoryRecord.hpp>
#include <polarai/backend/generic/runtime/Device.hpp>
#include <polar_backend_generic_export.h>

#include <list>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace polarai::backend::generic {

enum class MemoryDomain { Swap, RAM, Device };

struct LockDescriptor {
  core::internal::LockType lockType;
  MemoryDomain domain;
  Device* device;
};

/// LayerAllocator implements layered allocation strategy for LLVM backend.
///
/// Tensors can be allocated straight on device. However, they'll be moved to
/// more latent memory type if device does not have enough space to execute
/// other kernels.
///
/// Memory movements strategies:
/// 1. There are three maps indicating memory allocations for each memory domain
/// 2. Whenever runtime tries to perform a lock on a certain memory domain,
///    check if this Tensor is already locked. If it is, terminate.
/// 3. Otherwise, allocate memory on target device and copy data from low level
///    memory domain.
/// 4. If one of the allocation layers is out of memory, a callback function is
///    called to free up some space and copy data to more latent memory.
class POLAR_BACKEND_GENERIC_EXPORT LayerAllocator : public BackendAllocator {
private:
  std::mutex mMutex;

  std::unordered_map<std::string, std::shared_ptr<AllocatorLayerBase>>
      mDeviceAllocators;
  std::unordered_map<std::string, Device*> mDevices;
  std::unique_ptr<AllocatorLayerBase> mRAMAllocator;

  // fixme unordered_map and list are probably bad performers for this job.
  std::unordered_map<MemoryRecord, std::list<LockDescriptor>> mLocks;

  std::unordered_map<MemoryRecord, int> mMemTags;

  void updateHost(MemoryRecord record);

public:
  LayerAllocator() : mRAMAllocator(std::make_unique<TrivialAllocator>()) {}
  ~LayerAllocator() override = default;
  void registerDevice(Device& device) override;

  void allocate(const core::internal::TensorInternal& tensor,
                Device& device) override;
  void allocate(const MemoryRecord& record, Device& device) override;
  void allocate(const core::internal::TensorInternal& tensor) override;
  void allocate(const MemoryRecord& record) override;

  void deallocate(const core::internal::TensorInternal& tensor) override;

  void lock(const core::internal::TensorInternal& tensor,
            core::internal::LockType type) override;
  void lock(const MemoryRecord& record,
            core::internal::LockType type) override;
  void lock(const core::internal::TensorInternal& tensor, Device& device,
            core::internal::LockType type) override;
  void lock(const MemoryRecord& tensor, Device& device,
            core::internal::LockType type) override;

  void release(const core::internal::TensorInternal& tensor) override;
  void release(const MemoryRecord& tensor) override;
  void release(const core::internal::TensorInternal& tensor, Device& device);
  void release(const MemoryRecord& record, Device& device) override;

  void* get(const core::internal::TensorInternal& tensor) override;
  void* get(const MemoryRecord& record) override;
  using BackendAllocator::get;

protected:
  void* getImpl(const core::internal::TensorInternal& tensor,
                Device& device) override;
  void* getImpl(const MemoryRecord& record, Device& device) override;
};
} // namespace polarai::backend::llvm
