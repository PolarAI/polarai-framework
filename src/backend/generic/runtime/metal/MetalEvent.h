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

#include <athena/backend/llvm/runtime/Event.h>
#include <athena/backend/llvm/runtime/runtime_export.h>

#import "Metal/Metal.h"

#include <future>
#include <vector>

namespace athena::backend::llvm {
class MetalDevice;
class ATH_RT_LLVM_EXPORT MetalEvent : public Event {
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
} // namespace athena::backend::llvm
