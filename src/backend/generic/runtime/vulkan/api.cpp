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

#include "VulkanContext.hpp"

#include <polarai/backend/generic/runtime/api.h>
#include <polar_rt_vulkan_export.h>

#include <vulkan/vulkan.h>

using namespace polarai::backend::generic;

extern "C" {

POLAR_RT_VULKAN_EXPORT auto initContext() -> Context* {
  VkApplicationInfo applicationInfo = {};
  applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  applicationInfo.pNext = nullptr;
  applicationInfo.pApplicationName = "PolarAI Framework";
  applicationInfo.applicationVersion = 0;
  applicationInfo.pEngineName = "polarai";
  applicationInfo.engineVersion = 0;
  applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

  VkInstanceCreateInfo instanceCreateInfo = {};
  instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCreateInfo.pNext = nullptr;
  instanceCreateInfo.flags = 0;
  instanceCreateInfo.pApplicationInfo = &applicationInfo;
  instanceCreateInfo.enabledLayerCount = 0;
  instanceCreateInfo.ppEnabledLayerNames = nullptr;
  instanceCreateInfo.enabledExtensionCount = 0;
  instanceCreateInfo.ppEnabledExtensionNames = nullptr;

  VkInstance instance{VK_NULL_HANDLE};
  vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
  return new VulkanContext(instance);
}

POLAR_RT_VULKAN_EXPORT void closeContext(Context* ctx) { delete ctx; }
}
