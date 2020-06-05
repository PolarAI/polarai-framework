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

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <athena/backend/llvm/runtime/Event.h>
#include <athena/backend/llvm/runtime/runtime_export.h>

namespace athena::backend::llvm {
class OpenCLDevice;
void eventCallback(cl_event event, cl_int eventCommandStatus, void* userData);

class ATH_RT_LLVM_EXPORT OpenCLEvent final : public Event {
public:
  explicit OpenCLEvent(OpenCLDevice* dev, cl_event evt);

  void wait() override;

  void addCallback(std::function<void()> callback) override {
    mCallbacks.push_back(std::move(callback));
  }

  auto getNativeEvent() -> cl_event& { return mEvent; }

  auto getDevice() -> Device* override;

private:
  friend void eventCallback(cl_event event, cl_int eventCommandStatus,
                            void* userData);
  OpenCLDevice* mDevice;
  cl_event mEvent;
  std::vector<std::function<void()>> mCallbacks;
};
} // namespace athena::backend::llvm
