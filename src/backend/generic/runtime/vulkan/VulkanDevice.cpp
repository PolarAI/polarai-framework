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

#include "VulkanDevice.hpp"
#include "../utils/utils.hpp"
#include "VulkanAllocator.hpp"
#include "VulkanEvent.hpp"
#include "utils.hpp"

#include <polarai/backend/generic/runtime/Event.hpp>
#include <polarai/backend/generic/runtime/LaunchCommand.h>

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

namespace polarai::backend::generic {
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

  // Create DescriptorSetLayout
  std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;

  for (int i = 0; i < cmd.argsCount; i++) {
    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {};
    descriptorSetLayoutBinding.binding = i;
    descriptorSetLayoutBinding.descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding.descriptorCount = 1;
    descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    descriptorSetLayoutBindings.push_back(descriptorSetLayoutBinding);
  }

  // Create descriptor set layout
  // WARNING: all descriptor sets must be of the same type!
  VkDescriptorSetLayout descriptorSetLayout = {};
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};

  descriptorSetLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

  descriptorSetLayoutCreateInfo.bindingCount =
      descriptorSetLayoutBindings.size();
  descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();

  check(vkCreateDescriptorSetLayout(mDevice, &descriptorSetLayoutCreateInfo, 0,
                                    &descriptorSetLayout));

  // Create pipeline layout
  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
  pipelineLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.pNext = nullptr;
  pipelineLayoutCreateInfo.flags = 0;
  pipelineLayoutCreateInfo.setLayoutCount = 1;
  pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
  pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
  pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;
  VkPipelineLayout pipelineLayout;
  check(vkCreatePipelineLayout(mDevice, &pipelineLayoutCreateInfo, 0,
                               &pipelineLayout));

  // Create compute pipeline
  VkPipelineShaderStageCreateInfo stageInfo = {};
  stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stageInfo.pNext = nullptr;
  stageInfo.flags = 0;
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module = mShaderModule;
  stageInfo.pName = cmd.kernelName;
  stageInfo.pSpecializationInfo = nullptr;

  VkComputePipelineCreateInfo computePipelineCreateInfo = {};
  computePipelineCreateInfo.sType =
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computePipelineCreateInfo.pNext = nullptr;
  computePipelineCreateInfo.flags = 0;
  computePipelineCreateInfo.stage = stageInfo;
  computePipelineCreateInfo.layout = pipelineLayout;
  computePipelineCreateInfo.basePipelineHandle = nullptr;
  computePipelineCreateInfo.basePipelineIndex = 0;
  VkPipeline pipeline;
  check(vkCreateComputePipelines(mDevice, 0, 1, &computePipelineCreateInfo, 0,
                                 &pipeline));

  // Create descriptor pool
  VkDescriptorPoolSize descriptorPoolSize = {};
  descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptorPoolSize.descriptorCount = descriptorSetLayoutBindings.size();

  VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
  descriptorPoolCreateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptorPoolCreateInfo.pNext = nullptr;
  descriptorPoolCreateInfo.flags = 0;
  descriptorPoolCreateInfo.maxSets = 1;
  descriptorPoolCreateInfo.poolSizeCount = 1;
  descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
  VkDescriptorPool descriptorPool;
  check(vkCreateDescriptorPool(mDevice, &descriptorPoolCreateInfo, 0,
                               &descriptorPool));

  // Allocate descriptor sets
  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
  VkDescriptorSet descriptorSet;
  descriptorSetAllocateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.pNext = nullptr;
  descriptorSetAllocateInfo.descriptorPool = descriptorPool;
  descriptorSetAllocateInfo.descriptorSetCount = 1;
  descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;
  check(vkAllocateDescriptorSets(mDevice, &descriptorSetAllocateInfo,
                                 &descriptorSet));

  // Set write descriptors
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
    writeDescriptorSet.dstSet = descriptorSet;
    writeDescriptorSet.dstBinding = i;
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet.pBufferInfo = &info;

    vkUpdateDescriptorSets(mDevice, 1, &writeDescriptorSet, 0, nullptr);
  }

  // Create command pool
  VkCommandPoolCreateInfo commandPoolCreateInfo = {};
  commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolCreateInfo.pNext = nullptr;
  commandPoolCreateInfo.flags = 0;
  commandPoolCreateInfo.queueFamilyIndex = mQueueFamilyIndex;
  VkCommandPool commandPool;
  check(vkCreateCommandPool(mDevice, &commandPoolCreateInfo,
                            /*pAllocator=*/nullptr, &commandPool));

  // Create compute command buffer
  VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
  commandBufferAllocateInfo.sType =
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  commandBufferAllocateInfo.pNext = nullptr;
  commandBufferAllocateInfo.commandPool = commandPool;
  commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  commandBufferAllocateInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  check(vkAllocateCommandBuffers(mDevice, &commandBufferAllocateInfo,
                                 &commandBuffer));

  VkCommandBufferBeginInfo commandBufferBeginInfo = {};
  commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  commandBufferBeginInfo.pNext = nullptr;
  commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  commandBufferBeginInfo.pInheritanceInfo = nullptr;

  check(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelineLayout, 0, 1, &descriptorSet, 0, 0);

  auto& kernelDesc = mSpvModule->kernels[cmd.kernelName];
  size_t groupsX = kernelDesc.globalX / kernelDesc.localX;
  size_t groupsY = kernelDesc.globalY / kernelDesc.localY;
  size_t groupsZ = kernelDesc.globalZ / kernelDesc.localZ;
  vkCmdDispatch(commandBuffer, groupsX, groupsY, groupsZ);
  check(vkEndCommandBuffer(commandBuffer));

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.pNext = nullptr;
  submitInfo.waitSemaphoreCount = 0;
  submitInfo.pWaitSemaphores = 0;
  submitInfo.pWaitDstStageMask = 0;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  submitInfo.signalSemaphoreCount = 0;
  submitInfo.pSignalSemaphores = nullptr;

  VkFence fence;
  VkFenceCreateInfo fenceCreateInfo = {};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.flags = 0;
  check(vkCreateFence(mDevice, &fenceCreateInfo, nullptr, &fence));

  check(vkQueueSubmit(mQueue, 1, &submitInfo, fence));

  // check(vkWaitForFences(mDevice, 1, &fence, VK_TRUE, 10000000000000));
  return new VulkanEvent(this, fence);
}
} // namespace polarai::backend::llvm
