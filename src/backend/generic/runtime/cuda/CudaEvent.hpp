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

#include "CudaDevice.hpp"
#include <polarai/backend/generic/runtime/Event.hpp>

#include <cuda.h>

#include <future>

namespace polarai::backend::generic {
class CudaDevice;
class CudaEvent final : public Event {
public:
  explicit CudaEvent(CudaDevice* device, CUevent evt);
  ~CudaEvent() override;

  void wait() override;

  void addCallback(std::function<void()> callback) override {
    mCallbacks.push_back(std::move(callback));
  }

  auto getNativeEvent() -> CUevent& { return mEvent; }

  auto getDevice() -> Device* override;

private:
  CUevent mEvent;
  CudaDevice* mDevice;
  std::vector<std::function<void()>> mCallbacks;
};
} // namespace polarai::backend::llvm

