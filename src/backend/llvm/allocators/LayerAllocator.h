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

#ifndef ATHENA_LAYERALLOCATOR_H
#define ATHENA_LAYERALLOCATOR_H

#include "TrivialAllocator.h"

#include <athena/backend/llvm/AllocatorLayerBase.h>
#include <athena/backend/llvm/BackendAllocator.h>
#include <athena/backend/llvm/MemoryRecord.h>
#include <athena/backend/llvm/llvm_export.h>
#include <athena/backend/llvm/runtime/Device.h>

#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace athena::backend::llvm {

enum class MemoryDomain { Swap, RAM, Device };

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
class ATH_BACKEND_LLVM_EXPORT LayerAllocator : public BackendAllocator {
private:
  std::mutex mMutex;

  std::unordered_map<std::string, std::shared_ptr<AllocatorLayerBase>>
      mDeviceAllocators;
  std::unordered_map<MemoryRecord, MemoryDomain> mLockDomainMap;
  // todo replace map with another structure to allow memory on multiple devices
  std::unordered_map<MemoryRecord, Device*> mDeviceMap;
  std::unordered_set<MemoryRecord> mRAMSet;
  std::unordered_map<MemoryRecord, std::string> mSwapMap;

  std::unique_ptr<AllocatorLayerBase> mRAMAllocator;

  bool isAllocated(const MemoryRecord& record);

  MemoryDomain getAllocationDomain(const MemoryRecord& record);

public:
  LayerAllocator() : mRAMAllocator(std::make_unique<TrivialAllocator>()) {}
  ~LayerAllocator() override = default;
  void registerDevice(Device& device) override;
  void allocate(const core::inner::Tensor& tensor, Device& device) override;
  void lock(const core::inner::Tensor& tensor, Device& device) override;
  void allocate(const core::inner::Tensor& tensor) override;
  void allocate(MemoryRecord record);
  void deallocate(const core::inner::Tensor& tensor) override;
  void* get(const core::inner::Tensor& tensor) override;
  void lock(const core::inner::Tensor& tensor) override;
  void release(const core::inner::Tensor& tensor) override;
  using BackendAllocator::get;

protected:
  void* getImpl(const core::inner::Tensor& tensor, Device& device) override;
};
} // namespace athena::backend::llvm

#endif // ATHENA_LAYERALLOCATOR_H
