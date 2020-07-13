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

#include "MetalAllocator.hh"
#include <polarai/backend/generic/runtime/Device.hpp>

#import <Metal/Metal.h>

#include <string>

namespace polarai::backend::generic {
class MetalDevice : public Device {
public:
  MetalDevice(id<MTLDevice> device);
  auto getProvider() const -> DeviceProvider override {
    return DeviceProvider::METAL;
  }
  auto getKind() const -> DeviceKind override { return DeviceKind::GPU; }
  std::string getDeviceName() const override { return mDeviceName; };
  bool isPartitionSupported(PartitionDomain domain) override { return false; }
  bool hasAllocator() override { return true; }

  std::vector<std::shared_ptr<Device>>
  partition(PartitionDomain domain) override {
    return std::vector<std::shared_ptr<Device>>{};
  };
  std::shared_ptr<AllocatorLayerBase> getAllocator() override {
    return mAllocator;
  };

  bool operator==(const Device &device) const override {
    return mDeviceName == device.getDeviceName();
  };

  void copyToHost(MemoryRecord record, void *dest) const override {
    auto buf = *static_cast<id<MTLBuffer>*>(mAllocator->getPtr(record));

    void *sharedPtr = buf.contents;
    std::memcpy(dest, sharedPtr, record.allocationSize);
  };
  void copyToDevice(MemoryRecord record, void *src) const override {
    id<MTLBuffer> buf =
        *static_cast<id<MTLBuffer> *>(mAllocator->getPtr(record));
    void *sharedPtr = buf.contents;
    std::memcpy(sharedPtr, src, record.allocationSize);
  };

  Event *launch(BackendAllocator &, LaunchCommand &, Event *) override;

  void consumeEvent(Event *) override{};

  void
  selectBinary(std::vector<std::shared_ptr<ProgramDesc>> &programs) override;

private:
  id<MTLDevice> mDevice;
  std::shared_ptr<MetalAllocator> mAllocator;
  std::string mDeviceName;
  std::shared_ptr<ProgramDesc> mProgram;
  id<MTLLibrary> mLibrary;
};
} // namespace polarai::backend::llvm
