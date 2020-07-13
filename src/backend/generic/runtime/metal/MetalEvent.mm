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

#include "MetalEvent.hh"
#include "MetalDevice.hh"

namespace polarai::backend::generic {
MetalEvent::MetalEvent(MetalDevice* device, id<MTLCommandBuffer> cmdBuf)
    : mCmdBuf(cmdBuf), mDevice(device) {}

void MetalEvent::wait() {
  // todo is thread safety required here?
  [mCmdBuf waitUntilCompleted];
  for (auto& cb : mCallbacks) {
    cb();
  }
  mCallbacks.clear();
}
auto MetalEvent::getDevice() -> Device* { return mDevice; };
MetalEvent::~MetalEvent() { /* todo free buffer */ }
} // namespace polarai::backend::llvm
