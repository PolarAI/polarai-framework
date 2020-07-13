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

#include "VulkanDevice.hpp"
#include <polarai/backend/generic/runtime/Event.hpp>

#include <vulkan/vulkan.h>

namespace polarai::backend::generic {
class VulkanEvent final : public Event {
public:
  VulkanEvent(VulkanDevice* device, VkFence fence);
  ~VulkanEvent() override;

  void wait() override;

  void addCallback(std::function<void()> callback) override {
    mCallbacks.push_back(std::move(callback));
  }

  auto getNativeEvent() -> VkFence& { return mFence; }

  auto getDevice() -> Device* override;

private:
  VkFence mFence;
  VulkanDevice* mDevice;
  std::vector<std::function<void()>> mCallbacks;
};
} // namespace polarai::backend::llvm


