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

#include <polarai/backend/generic/runtime/Event.hpp>

#import "Metal/Metal.h"

#include <future>
#include <vector>

namespace polarai::backend::generic {
class MetalDevice;
class MetalEvent : public Event {
public:
  explicit MetalEvent(MetalDevice* device, id<MTLCommandBuffer> cmdBuf);
  ~MetalEvent() override;

  void wait() override;

  void addCallback(std::function<void()> callback) override {
    mCallbacks.push_back(std::move(callback));
  }

  auto getDevice() -> Device* override;

private:
  id<MTLCommandBuffer> mCmdBuf;
  MetalDevice* mDevice;
  std::vector<std::function<void()>> mCallbacks;
};
} // namespace polarai::backend::llvm
