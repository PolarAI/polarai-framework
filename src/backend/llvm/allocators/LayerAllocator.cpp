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

#include "LayerAllocator.h"
#include <athena/utils/error/FatalError.h>

#include <mutex>

using namespace athena::utils;

namespace athena::backend::llvm {
void LayerAllocator::allocate(const core::internal::TensorInternal& tensor,
                              Device& device) {
  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};
  // Double allocation is noop.
  if (mDeviceAllocators[device.getDeviceName()]->isAllocated(record))
    return;

  mDeviceAllocators[device.getDeviceName()]->allocate(record);
  mLocks.insert({record, std::list<LockDescriptor>()});
  mMemTags[record] = 1;
}
void LayerAllocator::lock(const core::internal::TensorInternal& tensor,
                          Device& device, core::internal::LockType lockType) {
  std::scoped_lock curLock{mMutex};

  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};

  if (lockType == core::internal::LockType::READ_WRITE &&
      !mLocks[record].empty()) {
    new FatalError(
        ATH_BAD_ACCESS,
        "Attempt get READ_WRITE lock for tensor that is already locked: ",
        record.virtualAddress);
  }

  // fixme ensure device does not have another READ lock for this record.

  if (!mDeviceAllocators[device.getDeviceName()]->isAllocated(record)) {
    mDeviceAllocators[device.getDeviceName()]->allocate(record);
  }

  // This means that the device does not have the up to date data.
  if (mDeviceAllocators[device.getDeviceName()]->getTag(record) !=
      mMemTags[record]) {
    // fixme some devices may allow transfer without involving host.

    // Neither host has the up to date state
    if (mMemTags[record] != mRAMAllocator->getTag(record)) {
      updateHost(record);
    }

    device.copyToDevice(record, mRAMAllocator->getPtr(record));
  }

  if (lockType == core::internal::LockType::READ_WRITE) {
    mMemTags[record]++;
  }

  // Device is guaranteed to have up-to-date record now.
  mDeviceAllocators[device.getDeviceName()]->setTag(record, mMemTags[record]);

  mLocks[record].emplace_back(
      LockDescriptor{lockType, MemoryDomain::Device, &device});
  mDeviceAllocators[device.getDeviceName()]->lock(record);
}
void LayerAllocator::allocate(const core::internal::TensorInternal& tensor) {
  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};
  allocate(record);
}
void LayerAllocator::allocate(MemoryRecord record) {
  if (mRAMAllocator->isAllocated(record))
    return;

  mRAMAllocator->allocate(record);
  mLocks.insert({record, std::list<LockDescriptor>()});
  mMemTags[record] = 1;
}
void LayerAllocator::deallocate(const core::internal::TensorInternal& tensor) {
  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};

  std::scoped_lock curLock{mMutex};

  if (!mLocks[record].empty()) {
    new FatalError(ATH_BAD_ACCESS, "Attempt to deallocate memory in use: ",
                   record.virtualAddress);
  }

  for (auto allocator : mDeviceAllocators) {
    if (allocator.second->isAllocated(record)) {
      allocator.second->deallocate(record);
    }
  }

  if (mRAMAllocator->isAllocated(record)) {
    mRAMAllocator->deallocate(record);
  }

  // todo release swap memory

  if (mMemTags.count(record) > 0) {
    mMemTags.erase(record);
  }

  if (mLocks.count(record)) {
    mLocks.erase(record);
  }
}
void* LayerAllocator::get(const core::internal::TensorInternal& tensor) {
  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};

  if (mRAMAllocator->isAllocated(record)) {
    return mRAMAllocator->getPtr(record);
  }

  new FatalError(ATH_BAD_ACCESS, "No host pointer for vaddr ",
                 record.virtualAddress);
  return nullptr; // suppress GCC warning
}

void LayerAllocator::lock(const core::internal::TensorInternal& tensor,
                          LockType type) {
  std::scoped_lock lock{mMutex};

  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};

  if (type == core::internal::LockType::READ_WRITE && !mLocks[record].empty()) {
    new FatalError(
        ATH_BAD_ACCESS,
        "Attempt get READ_WRITE lock for tensor that is already locked: ",
        record.virtualAddress);
  }

  if (!mRAMAllocator->isAllocated(record)) {
    mRAMAllocator->allocate(record);
  }

  if (mMemTags[record] != mRAMAllocator->getTag(record)) {
    updateHost(record);
  }

  if (type == core::internal::LockType::READ_WRITE) {
    mMemTags[record]++;
  }

  mRAMAllocator->setTag(record, mMemTags[record]);

  mLocks[record].push_back(LockDescriptor{type, MemoryDomain::RAM, nullptr});
  mRAMAllocator->lock(record);
}
void LayerAllocator::release(const core::internal::TensorInternal& tensor) {
  std::scoped_lock lock{mMutex};

  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};

  auto it = std::find_if(mLocks[record].begin(), mLocks[record].end(),
                         [&](const LockDescriptor& desc) {
                           return desc.domain == MemoryDomain::RAM;
                         });

  if (it != mLocks[record].end()) {
    mRAMAllocator->release(record);
    mLocks[record].erase(it);
  }
}
void* LayerAllocator::getImpl(const core::internal::TensorInternal& tensor,
                              Device& device) {
  std::scoped_lock lock{mMutex};
  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};

  if (mDeviceAllocators[device.getDeviceName()]->isAllocated(record)) {
    return mDeviceAllocators[device.getDeviceName()]->getPtr(record);
  }

  new FatalError(ATH_BAD_ACCESS, "Tensor ", record.virtualAddress,
                 " has no allocation on device ", device.getDeviceName());
  return nullptr; // suppress warnings.
}
void LayerAllocator::registerDevice(Device& device) {
  std::scoped_lock lock{mMutex};
  if (mDeviceAllocators.count(device.getDeviceName()) == 0) {
    if (device.hasAllocator()) {
      mDeviceAllocators[device.getDeviceName()] = device.getAllocator();
    } else {
      mDeviceAllocators[device.getDeviceName()] =
          std::move(std::shared_ptr<AllocatorLayerBase>(
              mRAMAllocator.get(), [](AllocatorLayerBase*) {}));
    }
    mDevices[device.getDeviceName()] = &device;
    auto& allocator = mDeviceAllocators[device.getDeviceName()];
    allocator->registerMemoryOffloadCallback(
        [this, &device](MemoryRecord record, AllocatorLayerBase& layer) {
          // Allocators have the same state of data. No need to copy.
          if (mRAMAllocator->getTag(record) == layer.getTag(record))
            return;
          if (!mRAMAllocator->isAllocated(record)) {
            mRAMAllocator->allocate(record);
          }
          void* hostPtr = mRAMAllocator->getPtr(record);

          device.copyToHost(record, hostPtr);
          mRAMAllocator->setTag(record, layer.getTag(record));
        });
  }
}
void LayerAllocator::updateHost(MemoryRecord record) {
  for (const auto& alloc : mDeviceAllocators) {
    if (alloc.second->getTag(record) == mMemTags[record] &&
        alloc.second->getTag(record) > 0) {
      if (!mRAMAllocator->isAllocated(record)) {
        mRAMAllocator->allocate(record);
      }

      mDevices[alloc.first]->copyToHost(record, mRAMAllocator->getPtr(record));
      break;
    }
  }
}
void LayerAllocator::release(const core::internal::TensorInternal& tensor,
                             Device& device) {
  std::scoped_lock lock{mMutex};

  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};

  auto it = std::find_if(mLocks[record].begin(), mLocks[record].end(),
                         [&](const LockDescriptor& desc) {
                           return desc.domain == MemoryDomain::Device &&
                                  desc.device->getDeviceName() ==
                                      device.getDeviceName();
                         });

  if (it != mLocks[record].end()) {
    mDeviceAllocators[device.getDeviceName()]->release(record);
    mLocks[record].erase(it);
  }
}
} // namespace athena::backend::llvm
