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

#include "OpenClDevice.h"
#include "OpenClEvent.h"
#include "athena/backend/llvm/runtime/TensorInfo.h"
#include "../utils/utils.h"

#include <athena/backend/llvm/BackendAllocator.h>
#include <athena/backend/llvm/runtime/LaunchCommand.h>

namespace athena::backend::llvm {
auto OpenCLDevice::launch(BackendAllocator& allocator, LaunchCommand& cmd,
                          Event* event) -> Event* {

  auto oclEvent = static_cast<OpenCLEvent*>(event);

  cl_int err;

  // todo check errors
  cl_kernel kernel =
      clCreateKernel(getLinkedProgram(), cmd.kernelName, &err);
  if (err != CL_SUCCESS) {
    std::terminate();
  }

  for (size_t i = 0; i < cmd.argsCount; i++) {
    if (cmd.args[i].type == ArgDesc::TENSOR) {
      auto tensor =
          static_cast<TensorInfo*>(cmd.args[i].arg);
      auto record = tensorInfoToRecord(tensor);
      auto* buf = allocator.get<cl_mem>(record, *this);
      clSetKernelArg(kernel, i, sizeof(cl_mem), buf);
    } else {
      clSetKernelArg(kernel, i, cmd.args[i].size, cmd.args[i].arg);
    }
  }

  cl_event* evt = nullptr;
  cl_event outEvent;
  cl_uint evtCount = 0;

  if (oclEvent) {
    evt = &oclEvent->getNativeEvent();
    evtCount = 1;
  }

  // todo check errors.
  err = clEnqueueNDRangeKernel(mQueue->getNativeQueue(), kernel, cmd.workDim,
                         nullptr, // global offset
                         cmd.globalSize, cmd.globalSize,    // TODO second argument to nullptr
                         evtCount, // num events in wait list
                         evt,      // event list
                         &outEvent  // event
  );


  // FIXME hacky hack to avoid early memory freeing.
  clWaitForEvents(1, &outEvent);

  if (err != CL_SUCCESS) {
    std::terminate();
  }

  return new OpenCLEvent(outEvent);
}
} // namespace athena::backend::llvm
