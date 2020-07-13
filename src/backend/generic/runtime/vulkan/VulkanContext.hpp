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

#include <polarai/backend/generic/runtime/Context.hpp>

#include <vulkan/vulkan.h>

namespace polarai::backend::generic {
class VulkanContext : public Context {
public:
  VulkanContext(VkInstance instance);
  std::vector<std::shared_ptr<Device>>& getDevices() override;

private:
  std::vector<std::shared_ptr<Device>> mDevices;
  VkInstance mInstance; 
};
} // namespace polarai::backend::llvm
