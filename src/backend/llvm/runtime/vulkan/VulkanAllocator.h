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

#include "utils.hpp"
#include <athena/backend/llvm/AllocatorLayerBase.h>
#include <athena/backend/llvm/MemoryRecord.h>
#include <athena/utils/error/FatalError.h>

#include <vulkan/vulkan.h>

#include <unordered_map>
#include <unordered_set>

namespace athena::backend::llvm {
class VulkanAllocator : public AllocatorLayerBase {
public:
  struct MemDescriptor {
    VkBuffer buffer;
    VkDeviceMemory memory;
  };
private:
  size_t mStackPointer = std::numeric_limits<size_t>::max();
  VkPhysicalDevice mPhysicalDevice;
  VkDevice mDevice;
  MemoryOffloadCallbackT mOffloadCallback;
  std::unordered_map<MemoryRecord, MemDescriptor> mMemMap;
  std::unordered_set<MemoryRecord> mLockedAllocations;
  std::unordered_set<MemoryRecord> mReleasedAllocations;
  std::unordered_map<MemoryRecord, int> mTags;

  uint32_t findMemoryType(uint32_t memoryTypeBits,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memoryProperties;

    vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
      if ((memoryTypeBits & (1 << i)) &&
          ((memoryProperties.memoryTypes[i].propertyFlags & properties) ==
           properties))
        return i;
    }
    return -1;
  }

  bool createMemDesc(MemDescriptor& desc, size_t size) {
    VkBufferCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.flags = 0;
    createInfo.size = size;
    createInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;
    createInfo.pQueueFamilyIndices = nullptr;

    check(vkCreateBuffer(mDevice, &createInfo, nullptr, &desc.buffer));

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(mDevice, desc.buffer, &memoryRequirements);

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;

    allocateInfo.memoryTypeIndex =
        findMemoryType(memoryRequirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    check(vkAllocateMemory(mDevice, &allocateInfo, nullptr, &desc.memory));

    check(vkBindBufferMemory(mDevice, desc.buffer, desc.memory, 0));

    return true;
  }

  void freeMemory(MemoryRecord record) {
    // NOOP at the moment
  }

public:
  VulkanAllocator(VkPhysicalDevice pDevice, VkDevice device)
      : mPhysicalDevice(pDevice), mDevice(device){};
  // todo free all resources
  ~VulkanAllocator() override = default;

  void registerMemoryOffloadCallback(MemoryOffloadCallbackT function) override {
  }
  void allocate(MemoryRecord record) override {
    if (mMemMap.count(record))
      return; // no double allocations are allowed

    MemDescriptor mem;
    // todo check allocation was successful
    createMemDesc(mem, record.allocationSize);

    mMemMap[record] = mem;
    mTags[record] = 1;
  }
  void deallocate(MemoryRecord record) override {
    if (mLockedAllocations.count(record)) {
      std::terminate();
    }

    // todo actually deallocate memory

    if (mReleasedAllocations.count(record)) {
      mReleasedAllocations.erase(record);
    }
    mTags[record] = 0;
  }
  void lock(MemoryRecord record) override { mLockedAllocations.insert(record); }
  void release(MemoryRecord record) override {
    mLockedAllocations.erase(record);
    mReleasedAllocations.insert(record);
  }

  void* getPtr(MemoryRecord record) override { return &mMemMap[record]; }

  bool isAllocated(const MemoryRecord& record) const override {
    return mMemMap.count(record) > 0;
  }

  size_t getTag(MemoryRecord record) override { return mTags[record]; }

  void setTag(MemoryRecord record, size_t tag) override { mTags[record] = tag; }

  MemDescriptor allocateStack(size_t size) {
    mStackPointer -= size;
    MemoryRecord record{mStackPointer, size};
    allocate(record);
    return mMemMap[record];
  }
};
} // namespace athena::backend::llvm
