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

#include "CudaAllocator.h"
#include <athena/backend/llvm/runtime/Device.h>

#include <cuda.h>
#include <string>

namespace athena::backend::llvm {
class CudaDevice : public Device {
public:
  CudaDevice(CUdevice device);

  ~CudaDevice() override;

  [[nodiscard]] auto getProvider() const -> DeviceProvider override {
    return DeviceProvider::CUDA;
  }
  [[nodiscard]] auto getKind() const -> DeviceKind override {
    return DeviceKind::GPU;
  }
  [[nodiscard]] auto getDeviceName() const -> std::string override;
  auto isPartitionSupported(PartitionDomain domain) -> bool override {
    return false;
  }
  auto hasAllocator() -> bool override { return true; }

  auto partition(PartitionDomain domain)
      -> std::vector<std::shared_ptr<Device>> override {
    return std::vector<std::shared_ptr<Device>>{};
  };
  auto getAllocator() -> std::shared_ptr<AllocatorLayerBase> override {
    return mAllocator;
  };

  bool operator==(const Device& device) const override {
    return mDeviceName == device.getDeviceName();
  };

  void copyToHost(const core::internal::TensorInternal& tensor,
                  void* dest) const override {
    MemoryRecord record{tensor.getVirtualAddress(),
                        tensor.getShapeView().getTotalSize() *
                            core::sizeOfDataType(tensor.getDataType())};
    copyToHost(record, dest);
  };
  void copyToHost(MemoryRecord record, void* dest) const override {
    auto* buf = reinterpret_cast<CUdeviceptr*>(mAllocator->getPtr(record));
    cuCtxSetCurrent(mDeviceContext);
    check(cuMemcpyDtoH(dest, *buf, record.allocationSize));
  };
  void copyToDevice(const core::internal::TensorInternal& tensor,
                    void* src) const override {
    MemoryRecord record{tensor.getVirtualAddress(),
                        tensor.getShapeView().getTotalSize() *
                            core::sizeOfDataType(tensor.getDataType())};
    copyToDevice(tensor, src);
  };
  void copyToDevice(MemoryRecord record, void* src) const override {
    auto* buf = reinterpret_cast<CUdeviceptr*>(mAllocator->getPtr(record));
    cuCtxSetCurrent(mDeviceContext);
    check(cuMemcpyHtoD(*buf, src, record.allocationSize));
  };

  auto launch(BackendAllocator&, LaunchCommand&, Event*) -> Event* override;

  void consumeEvent(Event* evt) override;

  void
  selectBinary(std::vector<std::shared_ptr<ProgramDesc>>& programs) override;

  auto getDeviceContext() -> CUcontext { return mDeviceContext; }

private:
  CUdevice mDevice;
  CUcontext mDeviceContext;
  CUstream mStream;
  std::shared_ptr<ProgramDesc> mPtxModule;
  CUmodule mMainModule;
  std::string mDeviceName;
  std::shared_ptr<AllocatorLayerBase> mAllocator;
};
} // namespace athena::backend::llvm
