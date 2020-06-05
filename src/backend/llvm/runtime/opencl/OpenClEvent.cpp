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

#include "OpenClEvent.h"
#include "OpenClDevice.h"

namespace athena::backend::llvm {
void eventCallback(cl_event event, cl_int eventCommandStatus, void* userData) {
  auto* athEvent = static_cast<athena::backend::llvm::OpenCLEvent*>(userData);
  if (eventCommandStatus == CL_COMPLETE) {
    for (auto& callback : athEvent->mCallbacks) {
      callback();
    }
  }
}
OpenCLEvent::OpenCLEvent(OpenCLDevice* device, cl_event evt)
    : mDevice(device), mEvent(evt) {
  clSetEventCallback(evt, CL_COMPLETE, eventCallback, this);
};
void OpenCLEvent::wait() { clWaitForEvents(1, &mEvent); }
auto OpenCLEvent::getDevice() -> Device* { return mDevice; };
} // namespace athena::backend::llvm
