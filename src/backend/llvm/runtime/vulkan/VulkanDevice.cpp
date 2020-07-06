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

#include "VulkanDevice.h"
#include "../utils/utils.h"
#include "VulkanAllocator.h"
#include "VulkanEvent.h"
#include "utils.hpp"

#include <athena/backend/llvm/runtime/Event.h>
#include <athena/backend/llvm/runtime/LaunchCommand.h>

#include <vulkan/vulkan.h>

static uint32_t getComputeQueueFamilyIndex(VkPhysicalDevice physicalDevice) {
  uint32_t queueFamilyCount;

  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           queueFamilies.data());

  uint32_t i = 0;
  for (; i < queueFamilies.size(); ++i) {
    VkQueueFamilyProperties props = queueFamilies[i];

    if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      break;
    }
  }

  if (i == queueFamilies.size()) {
    std::terminate(); // todo no queue families with compute support, throw.
  }

  return i;
}

namespace athena::backend::llvm {
VulkanDevice::VulkanDevice(VkPhysicalDevice device) : mPhysicalDevice(device) {
  VkDeviceQueueCreateInfo queueCreateInfo = {};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  mQueueFamilyIndex = getComputeQueueFamilyIndex(mPhysicalDevice);
  queueCreateInfo.queueFamilyIndex = mQueueFamilyIndex;
  queueCreateInfo.queueCount = 1;
  float queuePriorities = 1.0;
  queueCreateInfo.pQueuePriorities = &queuePriorities;

  VkDeviceCreateInfo deviceCreateInfo = {};

  VkPhysicalDeviceFeatures deviceFeatures = {};

  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.enabledLayerCount = 0;
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

  vkCreateDevice(mPhysicalDevice, &deviceCreateInfo, nullptr, &mDevice);
  vkGetDeviceQueue(mDevice, mQueueFamilyIndex, 0, &mQueue);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(mPhysicalDevice, &props);
  mDeviceName = props.deviceName;

  mAllocator = std::make_shared<VulkanAllocator>(mPhysicalDevice, mDevice);
}

std::string VulkanDevice::getDeviceName() const { return mDeviceName; }

void VulkanDevice::selectBinary(
    std::vector<std::shared_ptr<ProgramDesc>>& programs) {
  for (auto& prog : programs) {
    if (prog->type == ProgramDesc::Type::SPIRV_SHADER) {
      mSpvModule = prog;
      break;
    }
  }

  VkShaderModuleCreateInfo moduleCreateInfo = {};
  moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  moduleCreateInfo.pNext = nullptr;
  moduleCreateInfo.flags = 0;
  moduleCreateInfo.codeSize = mSpvModule->data.size();
  moduleCreateInfo.pCode = reinterpret_cast<uint32_t*>(mSpvModule->data.data());
  check(vkCreateShaderModule(mDevice, &moduleCreateInfo, nullptr,
                             &mShaderModule));
}

Event* VulkanDevice::launch(BackendAllocator& allocator, LaunchCommand& cmd,
                            Event* blockingEvent) {
  if (blockingEvent) {
    blockingEvent->wait();
  }

  std::vector<VkDescriptorSetLayoutBinding> bindings;

  for (int i = 0; i < cmd.argsCount; i++) {
    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {};
    descriptorSetLayoutBinding.binding = i;
    descriptorSetLayoutBinding.descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding.descriptorCount = 1;
    descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(descriptorSetLayoutBinding);
  }

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = bindings.size();
  descriptorSetLayoutCreateInfo.pBindings = bindings.data();

  VkDescriptorSetLayout layout;
  check(vkCreateDescriptorSetLayout(mDevice, &descriptorSetLayoutCreateInfo,
                                    nullptr, &layout));

  VkDescriptorPoolSize descriptorPoolSize = {};
  descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptorPoolSize.descriptorCount = bindings.size();

  VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
  descriptorPoolCreateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptorPoolCreateInfo.maxSets = 1;
  descriptorPoolCreateInfo.poolSizeCount = 1;
  descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;

  VkDescriptorPool descriptorPool;
  check(vkCreateDescriptorPool(mDevice, &descriptorPoolCreateInfo, nullptr,
                               &descriptorPool));

  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
  descriptorSetAllocateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.descriptorPool = descriptorPool;
  descriptorSetAllocateInfo.descriptorSetCount = 1;
  descriptorSetAllocateInfo.pSetLayouts = &layout;

  VkDescriptorSet descriptorSet;
  check(vkAllocateDescriptorSets(mDevice, &descriptorSetAllocateInfo,
                                 &descriptorSet));

  for (int i = 0; i < cmd.argsCount; i++) {
    VkDescriptorBufferInfo info = {};
    if (cmd.args[i].type == ArgDesc::TENSOR) {
      auto tensor = static_cast<TensorInfo*>(cmd.args[i].arg);
      auto record = tensorInfoToRecord(tensor);
      auto buf = allocator.get<VulkanAllocator::MemDescriptor>(record, *this);
      info.buffer = buf->buffer;
      info.offset = 0;
      info.range = record.allocationSize;
    } else {
      auto memdesc = mAllocator->allocateStack(cmd.args[i].size);
      void* hostPtr;
      vkMapMemory(mDevice, memdesc.memory, 0, cmd.args[i].size, 0, &hostPtr);
      memcpy(hostPtr, cmd.args[i].arg, cmd.args[i].size);
      vkUnmapMemory(mDevice, memdesc.memory);
      info.buffer = memdesc.buffer;
      info.offset = 0;
      info.range = cmd.args[i].size;
    }
    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = descriptorSet; // write to this descriptor set.
    writeDescriptorSet.dstBinding = i; // write to the first, and only binding.
    writeDescriptorSet.descriptorCount = 1; // update a single descriptor.
    writeDescriptorSet.descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
    writeDescriptorSet.pBufferInfo = &info;

    vkUpdateDescriptorSets(mDevice, 1, &writeDescriptorSet, 0, nullptr);
  }
  VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
  shaderStageCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStageCreateInfo.module = mShaderModule;
  shaderStageCreateInfo.pName = cmd.kernelName;

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
  pipelineLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.setLayoutCount = 1;
  pipelineLayoutCreateInfo.pSetLayouts = &layout;

  VkPipelineLayout pipelineLayout;
  check(vkCreatePipelineLayout(mDevice, &pipelineLayoutCreateInfo, nullptr,
                               &pipelineLayout));

  VkComputePipelineCreateInfo pipelineCreateInfo = {};
  pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCreateInfo.stage = shaderStageCreateInfo;
  pipelineCreateInfo.layout = pipelineLayout;

  VkPipeline pipeline;
  check(vkCreateComputePipelines(mDevice, VK_NULL_HANDLE, 1,
                                 &pipelineCreateInfo, nullptr, &pipeline));

  VkCommandPoolCreateInfo commandPoolCreateInfo = {};
  commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolCreateInfo.flags = 0;

  VkCommandPool commandPool;
  commandPoolCreateInfo.queueFamilyIndex = mQueueFamilyIndex;
  check(vkCreateCommandPool(mDevice, &commandPoolCreateInfo, nullptr,
                            &commandPool));

  VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
  commandBufferAllocateInfo.sType =
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  commandBufferAllocateInfo.commandPool = commandPool;

  commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  commandBufferAllocateInfo.commandBufferCount = 1;
  VkCommandBuffer commandBuffer;
  check(vkAllocateCommandBuffers(mDevice, &commandBufferAllocateInfo,
                                 &commandBuffer));

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(commandBuffer, &beginInfo));

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

  auto& kernelDesc = mSpvModule->kernels[cmd.kernelName];
  size_t groupsX = kernelDesc.globalX / kernelDesc.localX;
  size_t groupsY = kernelDesc.globalY / kernelDesc.localY;
  size_t groupsZ = kernelDesc.globalZ / kernelDesc.localZ;
  vkCmdDispatch(commandBuffer, groupsX, groupsY, groupsZ);
  check(vkEndCommandBuffer(commandBuffer));

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  VkFence fence;
  VkFenceCreateInfo fenceCreateInfo = {};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.flags = 0;
  check(vkCreateFence(mDevice, &fenceCreateInfo, nullptr, &fence));

  check(vkQueueSubmit(mQueue, 1, &submitInfo, fence));

  return new VulkanEvent(this, fence);
}
} // namespace athena::backend::llvm
