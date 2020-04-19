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
#include <athena/core/FatalError.h>

#include <mutex>

using namespace athena::core;

namespace athena::backend::llvm {
void LayerAllocator::allocate(const core::inner::Tensor& tensor,
                              Device& device) {
  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};
  if (isAllocated(record))
    return;

  mDeviceAllocators[device.getDeviceName()]->allocate(record);
  mDeviceMap.insert({record, &device});

  if (!device.hasAllocator()) {
    mRAMSet.insert(record);
  }
}
void LayerAllocator::lock(const core::inner::Tensor& tensor, Device& device) {
  std::scoped_lock curLock{mMutex};

  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};

  if (mLockDomainMap.count(record) > 0) {
    new FatalError(ATH_BAD_ACCESS,
                   "Tensor is already locked. vaddr:", record.virtualAddress);
  }

  if (!isAllocated(record)) {
    new FatalError(ATH_BAD_ACCESS,
                   "Tensor is not allocated. vaddr: ", record.virtualAddress);
  }

  auto allocatedDomain = getAllocationDomain(record);
  void* ramPtr = nullptr;

  if (allocatedDomain == MemoryDomain::Device &&
      *mDeviceMap[record] != device) {
    auto* from = mDeviceMap[record];
    if (mRAMSet.count(record)) {
      ramPtr = mRAMAllocator->getPtr(record);
    } else {
      mRAMAllocator->allocate(record);
      ramPtr = mRAMAllocator->getPtr(record);
    }

    from->copyToHost(tensor, ramPtr);
    from->getAllocator()->deallocate(record);
  }

  // todo handle swap memory

  if (allocatedDomain == MemoryDomain::RAM) {
    ramPtr = mRAMAllocator->getPtr(record);
  }

  if (ramPtr != nullptr) {
    device.copyToDevice(tensor, ramPtr);
  }

  mDeviceAllocators[device.getDeviceName()]->lock(record);
  mLockDomainMap[record] = MemoryDomain::RAM;
}
void LayerAllocator::allocate(const core::inner::Tensor& tensor) {
  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};
  allocate(record);
}
void LayerAllocator::allocate(MemoryRecord record) {
  if (isAllocated(record) && getAllocationDomain(record) == MemoryDomain::RAM)
    return;

  mRAMAllocator->allocate(record);
  mRAMSet.insert(record);
}
void LayerAllocator::deallocate(const core::inner::Tensor& tensor) {
  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};

  if (isAllocated(record)) {
    if (mDeviceMap.count(record) > 0) {
      mDeviceAllocators[mDeviceMap[record]->getDeviceName()]->deallocate(
          record);
      mDeviceMap.erase(record);
    }
    if (mRAMSet.count(record)) {
      mRAMAllocator->deallocate(record);
      mRAMSet.erase(record);
    }
    // todo deallocate for swap memory
  }
}
void* LayerAllocator::get(const core::inner::Tensor& tensor) {
  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};

  if (mRAMSet.count(record)) {
    return mRAMAllocator->getPtr(record);
  }
  new FatalError(ATH_BAD_ACCESS, "No host pointer for vaddr ",
                 record.virtualAddress);
  return nullptr; // suppress GCC warning
}

void LayerAllocator::lock(const core::inner::Tensor& tensor) {
  std::scoped_lock lock{mMutex};

  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};

  if (mLockDomainMap.count(record) > 0) {
    new FatalError(ATH_BAD_ACCESS,
                   "Tensor is already locked. vaddr: ", record.virtualAddress);
  }

  if (!isAllocated(record)) {
    new FatalError(ATH_BAD_ACCESS,
                   "Tensor is not allocated. vaddr: ", record.virtualAddress);
  }

  auto allocatedDomain = getAllocationDomain(record);

  if (allocatedDomain == MemoryDomain::Device) {
    auto* from = mDeviceMap[record];
    void* ramPtr = nullptr;
    if (mRAMSet.count(record)) {
      ramPtr = mRAMAllocator->getPtr(record);
    } else {
      mRAMAllocator->allocate(record);
      ramPtr = mRAMAllocator->getPtr(record);
    }

    from->copyToHost(tensor, ramPtr);
  }

  // todo handle swap memory

  mLockDomainMap[record] = MemoryDomain::RAM;
}
void LayerAllocator::release(const core::inner::Tensor& tensor) {
  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};

  auto domain = mLockDomainMap[record];
  if (domain == MemoryDomain::Device) {
    mDeviceAllocators[mDeviceMap[record]->getDeviceName()]->release(record);
  }

  mLockDomainMap.erase(record);
}
void* LayerAllocator::getImpl(const core::inner::Tensor& tensor,
                              Device& device) {
  MemoryRecord record{tensor.getVirtualAddress(),
                      tensor.getSize() *
                          core::sizeOfDataType(tensor.getDataType())};
  if (mDeviceMap.count(record) && *mDeviceMap[record] == device) {
    return mDeviceAllocators[device.getDeviceName()]->getPtr(record);
  }
  std::terminate();
}
void LayerAllocator::registerDevice(Device& device) {
  if (mDeviceAllocators.count(device.getDeviceName()) == 0) {
    if (device.hasAllocator()) {
      mDeviceAllocators[device.getDeviceName()] = device.getAllocator();
    } else {
      mDeviceAllocators[device.getDeviceName()] =
          std::move(std::shared_ptr<AllocatorLayerBase>(
              mRAMAllocator.get(), [](AllocatorLayerBase*) {}));
    }
    auto& allocator = mDeviceAllocators[device.getDeviceName()];
    allocator->registerMemoryOffloadCallback(
        [this, &device](MemoryRecord record) {
          if (this->mRAMSet.count(record) == 0) {
            this->allocate(record);
          }
          void* hostPtr = mRAMAllocator->getPtr(record);

          device.copyToHost(record, hostPtr);
        });
  }
}
bool LayerAllocator::isAllocated(const MemoryRecord& record) {
  return mDeviceMap.count(record) || mRAMSet.count(record) ||
         mSwapMap.count(record);
}
MemoryDomain LayerAllocator::getAllocationDomain(const MemoryRecord& record) {
  if (mDeviceMap.count(record))
    return MemoryDomain::Device;
  if (mRAMSet.count(record))
    return MemoryDomain::RAM;
  if (mSwapMap.count(record))
    return MemoryDomain::Swap;
  new FatalError(ATH_FATAL_OTHER, "Unknown allocation domain for vaddr ",
                 record.virtualAddress);
  return MemoryDomain::Swap; // suppress GCC warning
}
} // namespace athena::backend::llvm
