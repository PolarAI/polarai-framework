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

#include "CudaEvent.hpp"

#include <utility>

namespace polarai::backend::generic {
CudaEvent::CudaEvent(CudaDevice* device, CUevent evt)
    : mEvent(evt), mDevice(device) {}

void CudaEvent::wait() {
  // todo is thread safety required here?
  cuCtxSetCurrent(mDevice->getDeviceContext());
  check(cuEventSynchronize(mEvent));
  for (auto& cb : mCallbacks) {
    cb();
  }
  mCallbacks.clear();
}
auto CudaEvent::getDevice() -> Device* { return mDevice; };
CudaEvent::~CudaEvent() { cuEventDestroy(mEvent); }
} // namespace polarai::backend::generic
