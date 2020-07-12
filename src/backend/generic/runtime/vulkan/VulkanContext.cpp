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
#include "VulkanDevice.h"

namespace athena::backend::llvm {
VulkanContext::VulkanContext(VkInstance instance) : mInstance(instance) {
  uint32_t deviceCount;
  vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr);

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  for (auto device : devices) {
    mDevices.push_back(std::make_shared<VulkanDevice>(device));
  }
}
std::vector<std::shared_ptr<Device>>& VulkanContext::getDevices() {
  return mDevices;
}
} // namespace athena::backend::llvm
