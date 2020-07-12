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

#include "SYCLEvent.h"
#include "SYCLDevice.h"

#include <utility>

namespace athena::backend::llvm {
SYCLEvent::SYCLEvent(SYCLDevice* device, cl::sycl::event evt)
    : mDevice(device), mEvent(std::move(evt)) {}

void SYCLEvent::wait() {
  // todo is thread safety required here?
  mEvent.wait();
  for (auto& cb : mCallbacks) {
    cb();
  }
  mCallbacks.clear();
}
auto SYCLEvent::getDevice() -> Device* { return mDevice; };
} // namespace athena::backend::llvm
