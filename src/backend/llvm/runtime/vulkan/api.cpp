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

#include "VulkanContext.h"

#include <athena/backend/llvm/runtime/api.h>
#include <athena/backend/llvm/runtime/runtime_export.h>

#include <vulkan/vulkan.h>

using namespace athena::backend::llvm;

extern "C" {

ATH_RT_LLVM_EXPORT Context* initContext() {
  VkApplicationInfo applicationInfo = {};
  applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  applicationInfo.pNext = nullptr;
  applicationInfo.pApplicationName = "MLIR Vulkan runtime";
  applicationInfo.applicationVersion = 0;
  applicationInfo.pEngineName = "mlir";
  applicationInfo.engineVersion = 0;
  applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

  VkInstanceCreateInfo instanceCreateInfo = {};
  instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCreateInfo.pNext = nullptr;
  instanceCreateInfo.flags = 0;
  instanceCreateInfo.pApplicationInfo = &applicationInfo;
  instanceCreateInfo.enabledLayerCount = 0;
  instanceCreateInfo.ppEnabledLayerNames = 0;
  instanceCreateInfo.enabledExtensionCount = 0;
  instanceCreateInfo.ppEnabledExtensionNames = 0;

  VkInstance instance{VK_NULL_HANDLE};
  vkCreateInstance(&instanceCreateInfo, 0, &instance);
  // return new VulkanContext(instance);
  return nullptr;
}

ATH_RT_LLVM_EXPORT void closeContext(Context* ctx) { delete ctx; }
}