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

#pragma once

#include "BufferAllocator.h"
#include "SYCLQueue.h"
#include "USMAllocator.h"

#include <athena/backend/llvm/runtime/Device.h>
#include <athena/backend/llvm/runtime/runtime_export.h>

#include <CL/sycl.hpp>

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

namespace athena::backend::llvm {
class ATH_RT_LLVM_EXPORT SYCLDevice : public Device {
public:
  explicit SYCLDevice(cl::sycl::device device)
      : mRealDevice(std::move(device)) {

    mDeviceName = mRealDevice.get_info<cl::sycl::info::device::name>();

    mQueue = std::make_unique<SYCLQueue>(mRealDevice);
    if (mRealDevice.has_extension("cl_intel_unified_shared_memory_preview")) {
      mAllocator = std::make_shared<USMAllocator>(mQueue->getNativeQueue());
      mUsesUSM = true;
    } else {
      mAllocator = std::make_shared<BufferAllocator>(
          mQueue->getNativeQueue().get_context());
      mUsesUSM = false;
    }

    populateKernelMap();
  }

  [[nodiscard]] auto getDeviceName() const -> std::string override {
    return mDeviceName;
  }
  auto getProvider() const -> DeviceProvider override {
    return DeviceProvider::SYCL;
  }
  auto getKind() const -> DeviceKind override {
    if (mRealDevice.is_cpu() || mRealDevice.is_host()) {
      return DeviceKind::CPU;
    }
    if (mRealDevice.is_gpu()) {
      return DeviceKind::GPU;
    }
    return DeviceKind::OTHER_ACCELERATOR;
  }
  auto isPartitionSupported(PartitionDomain domain) -> bool override {
    return false; // todo implement
  }
  auto partition(PartitionDomain domain)
      -> std::vector<std::shared_ptr<Device>> override {
    return std::vector<std::shared_ptr<Device>>{}; // todo implement
  }
  auto hasAllocator() -> bool override { return true; }
  std::shared_ptr<AllocatorLayerBase> getAllocator() override {
    return mAllocator;
  }
  auto operator==(const Device& device) const -> bool override {
    // fixme it must compare device ids.
    return mDeviceName == device.getDeviceName();
  }
  void copyToHost(const core::internal::TensorInternal& tensor,
                  void* dest) const override {
    MemoryRecord record{tensor.getVirtualAddress(),
                        tensor.getShapeView().getTotalSize() *
                            core::sizeOfDataType(tensor.getDataType())};
    copyToHost(record, dest);
  }
  void copyToHost(MemoryRecord record, void* dest) const override {
    using namespace cl::sycl;
    if (mUsesUSM) {
      void* src = mAllocator->getPtr(record);
      auto evt =
          mQueue->getNativeQueue().memcpy(dest, src, record.allocationSize);
      evt.wait();
    } else {
      auto buf = *static_cast<buffer<char, 1>*>(mAllocator->getPtr(record));
      auto evt = mQueue->getNativeQueue().submit([&](handler& cgh) {
        auto acc = buf.get_access<access::mode::read>(cgh);
        cgh.copy(acc, static_cast<char*>(dest));
      });
      evt.wait();
    }
  }
  void copyToDevice(const core::internal::TensorInternal& tensor,
                    void* src) const override {
    MemoryRecord record{tensor.getVirtualAddress(),
                        tensor.getShapeView().getTotalSize() *
                            core::sizeOfDataType(tensor.getDataType())};
    copyToDevice(record, src);
  }
  void copyToDevice(MemoryRecord record, void* src) const override {
    using namespace cl::sycl;
    if (mUsesUSM) {
      void* dest = mAllocator->getPtr(record);
      auto evt =
          mQueue->getNativeQueue().memcpy(dest, src, record.allocationSize);
      evt.wait();
    } else {
      auto buf = *static_cast<buffer<char, 1>*>(mAllocator->getPtr(record));
      auto evt = mQueue->getNativeQueue().submit([&](handler& cgh) {
        auto acc = buf.get_access<access::mode::discard_write>(cgh);
        cgh.copy(static_cast<const char*>(src), acc);
      });
      evt.wait();
    }
  }

  auto getQueue() -> SYCLQueue& { return *mQueue; }
  auto getNativeDevice() -> cl::sycl::device { return mRealDevice; }

  auto launch(BackendAllocator&, LaunchCommand&, Event*) -> Event* override;

  void consumeEvent(Event*) override;

  void
  selectBinary(std::vector<std::shared_ptr<ProgramDesc>>& programs) override {};

private:
  using KernelFuncT = std::function<Event*(SYCLDevice*, BackendAllocator&,
                                           LaunchCommand&, Event*)>;
  void populateKernelMap();

  cl::sycl::device mRealDevice;
  std::unique_ptr<SYCLQueue> mQueue;
  std::shared_ptr<AllocatorLayerBase> mAllocator;

  bool mUsesUSM = false;

  std::string mDeviceName;
  std::unordered_map<std::string, KernelFuncT> mKernelMap;
};
} // namespace athena::backend::llvm
