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

#include "VulkanEvent.h"
#include "utils.hpp"

#include <utility>

namespace athena::backend::llvm {
VulkanEvent::VulkanEvent(VulkanDevice* device, VkFence fence)
    : mFence(fence), mDevice(device) {}

void VulkanEvent::wait() {
  // todo is thread safety required here?
  check(vkWaitForFences(mDevice->getVirtualDevice(), 1, &mFence, VK_TRUE,
                        10000000000000));
  for (auto& cb : mCallbacks) {
    cb();
  }
  mCallbacks.clear();
}
auto VulkanEvent::getDevice() -> Device* { return mDevice; };
VulkanEvent::~VulkanEvent() {
  vkDestroyFence(mDevice->getVirtualDevice(), mFence, nullptr);
}
} // namespace athena::backend::llvm
