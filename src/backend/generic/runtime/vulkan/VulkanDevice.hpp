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

#include "VulkanAllocator.hpp"

#include <polarai/backend/generic/runtime/Device.hpp>
#include <polarai/backend/generic/runtime/Event.hpp>

#include <string>
#include <vulkan/vulkan.h>

namespace polarai::backend::generic {
class VulkanDevice : public Device {
public:
  VulkanDevice(VkPhysicalDevice device);
  auto getProvider() const -> DeviceProvider override {
    return DeviceProvider::VULKAN;
  }
  auto getKind() const -> DeviceKind override { return DeviceKind::GPU; }
  std::string getDeviceName() const override;
  bool isPartitionSupported(PartitionDomain domain) override { return false; }
  bool hasAllocator() override { return true; }

  std::vector<std::shared_ptr<Device>>
  partition(PartitionDomain domain) override {
    return std::vector<std::shared_ptr<Device>>{};
  };
  std::shared_ptr<AllocatorLayerBase> getAllocator() override {
    return mAllocator;
  };

  bool operator==(const Device& device) const override {
    return mDeviceName == device.getDeviceName();
  };

  void copyToHost(MemoryRecord record, void* dest) const override {
    void* hostPtr;
    auto* buf = reinterpret_cast<VulkanAllocator::MemDescriptor*>(
        mAllocator->getPtr(record));
    vkMapMemory(mDevice, buf->memory, /*offset*/ 0, record.allocationSize,
                /*flags*/ 0, &hostPtr);
    memcpy(dest, hostPtr, record.allocationSize);
    vkUnmapMemory(mDevice, buf->memory);
  };
  void copyToDevice(MemoryRecord record, void* src) const override {
    void* hostPtr;
    auto* buf = reinterpret_cast<VulkanAllocator::MemDescriptor*>(
        mAllocator->getPtr(record));
    vkMapMemory(mDevice, buf->memory, /*offset*/ 0, record.allocationSize,
                /*flags*/ 0, &hostPtr);
    memcpy(hostPtr, src, record.allocationSize);
    vkUnmapMemory(mDevice, buf->memory);
  };

  Event* launch(BackendAllocator&, LaunchCommand&, Event*) override;

  void consumeEvent(Event* event) override{ delete event; };

  void
  selectBinary(std::vector<std::shared_ptr<ProgramDesc>>& programs) override;

  auto getVirtualDevice() { return mDevice; }

private:
  VkPhysicalDevice mPhysicalDevice;
  VkDevice mDevice;
  uint32_t mQueueFamilyIndex;
  VkQueue mQueue;
  std::shared_ptr<VulkanAllocator> mAllocator;
  std::shared_ptr<ProgramDesc> mSpvModule;
  VkShaderModule mShaderModule;
  std::string mDeviceName;
};
} // namespace polarai::backend::llvm
