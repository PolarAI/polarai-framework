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

#include <athena/backend/llvm/BackendAllocator.h>
#include <athena/backend/llvm/runtime/LaunchCommand.h>

namespace athena::backend::llvm {
auto OpenCLDevice::launch(BackendAllocator& allocator, LaunchCommand& cmd,
                          Event* event) -> Event* {

  // auto& queue = static_cast<OpenClQueue&>(oclDevice->getQueue());
  auto oclEvent = static_cast<OpenCLEvent*>(event);

  // todo check errors
  cl_kernel kernel =
      clCreateKernel(getLinkedProgram(), cmd.kernelName, nullptr);

  for (size_t i = 0; i < cmd.argsCount; i++) {
    if (cmd.args[i].type == ArgDesc::TENSOR) {
      auto tensor =
          static_cast<athena::core::internal::TensorInternal*>(cmd.args[i].arg);
      auto* buf = allocator.get<cl_mem>(*tensor, *this);
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
  auto err = clEnqueueNDRangeKernel(mQueue->getNativeQueue(), kernel, cmd.workDim,
                         nullptr, // global offset
                         cmd.globalSize, cmd.localSize,
                         evtCount, // num events in wait list
                         evt,      // event list
                         &outEvent  // event
  );

  if (err != CL_SUCCESS) {
    std::terminate();
  }

  return new OpenCLEvent(outEvent);
}

void OpenCLDevice::addModule(ProgramDesc prog) {
  cl_program program;
  switch (prog.type) {
  case ProgramDesc::ProgramType::TEXT:
    // fixme check errors
    program = clCreateProgramWithSource(mContext, 1, &prog.data, &prog.length,
                                        nullptr);
    break;
  case ProgramDesc::ProgramType::BINARY:
    program = clCreateProgramWithBinary(
        mContext, 1, &mClDeviceId, &prog.length,
        reinterpret_cast<const unsigned char**>(&prog.data), nullptr, nullptr);
    break;
  case ProgramDesc::ProgramType::SPIRV:
    // fixme OpenCL pre-2.1 uses clCreateProgramWithILKHR
    program = clCreateProgramWithIL(mContext, prog.data, prog.length, nullptr);
    break;
  }

  addProgram(program);
}

void OpenCLDevice::linkModules() {

  for (auto program : mPrograms) {
    // fixme should we wait for compilation to complete?
    clCompileProgram(program, 1, &mClDeviceId, nullptr, 0, nullptr, nullptr,
                     nullptr, nullptr);
  }

  auto prog =
      clLinkProgram(mContext, 1, &mClDeviceId, nullptr, mPrograms.size(),
                    mPrograms.data(), nullptr, nullptr, nullptr);
  setLinkedProgram(prog);
}
} // namespace athena::backend::llvm
