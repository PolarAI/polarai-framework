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

#include <athena/backend/llvm/runtime/api.h>
#include <athena/backend/llvm/runtime/runtime_export.h>

using namespace athena::backend::llvm;

extern "C" {
ATH_RT_LLVM_EXPORT DeviceContainer getAvailableDevices() {
  cl_uint platformCount;
  clGetPlatformIDs(0, nullptr, &platformCount);

  std::vector<cl_platform_id> platforms;
  platforms.resize(platformCount);

  clGetPlatformIDs(platformCount, platforms.data(), nullptr);

  std::vector<cl_device_id> allDevices;

  for (auto platform : platforms) {
    cl_uint numDevices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);

    std::vector<cl_device_id> devices;
    devices.resize(numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(),
                   nullptr);

    allDevices.insert(allDevices.end(), devices.begin(), devices.end());
  }

  std::allocator<OpenCLDevice> allocator;
  auto oclDevices = allocator.allocate(allDevices.size());

  int i = 0;
  for (auto device : allDevices) {
    new (oclDevices + i++) OpenCLDevice(device);
  }

  DeviceContainer deviceContainer{oclDevices, allDevices.size()};
  return deviceContainer;
}

ATH_RT_LLVM_EXPORT void addProgram(athena::backend::llvm::Device* device,
                                   ProgramDesc prog) {
  auto oclDevice = static_cast<OpenCLDevice*>(device);
  auto ctx = oclDevice->getContext();
  auto dev = oclDevice->getNativeDevice();

  cl_program program;
  switch (prog.type) {
  case ProgramDesc::ProgramType::TEXT:
    // fixme check errors
    program =
        clCreateProgramWithSource(ctx, 1, &prog.data, &prog.length, nullptr);
    break;
  case ProgramDesc::ProgramType::BINARY:
    program = clCreateProgramWithBinary(
        ctx, 1, &dev, &prog.length,
        reinterpret_cast<const unsigned char**>(&prog.data), nullptr, nullptr);
    break;
  case ProgramDesc::ProgramType::SPIRV:
    // fixme OpenCL pre-2.1 uses clCreateProgramWithILKHR
    program = clCreateProgramWithIL(ctx, prog.data, prog.length, nullptr);
    break;
  }

  oclDevice->addProgram(program);
}

ATH_RT_LLVM_EXPORT void linkPrograms(Device* device) {
  auto oclDevice = static_cast<OpenCLDevice*>(device);
  auto dev = oclDevice->getNativeDevice();
  auto ctx = oclDevice->getContext();

  for (auto program : oclDevice->getPrograms()) {
    // fixme should we wait for compilation to complete?
    clCompileProgram(program, 1, &dev, nullptr, 0, nullptr, nullptr, nullptr,
                     nullptr);
  }

  auto prog =
      clLinkProgram(ctx, 1, &dev, nullptr, oclDevice->getPrograms().size(),
                    oclDevice->getPrograms().data(), nullptr, nullptr, nullptr);
  oclDevice->setLinkedProgram(prog);
}
ATH_RT_LLVM_EXPORT void
launch(athena::backend::llvm::Device* device,
       athena::backend::llvm::BackendAllocator* backendAllocator,
       LaunchCommand cmd) {
  auto oclDevice = static_cast<OpenCLDevice*>(device);
  auto& queue = static_cast<OpenClQueue&>(oclDevice->getQueue());

  // todo check errors
  cl_kernel kernel =
      clCreateKernel(oclDevice->getLinkedProgram(), cmd.kernelName, nullptr);

  for (size_t i = 0; i < cmd.argsCount; i++) {
    if (cmd.args[i].type == ArgDesc::TENSOR) {
      auto tensor =
          static_cast<athena::core::internal::TensorInternal*>(cmd.args[i].arg);
      auto* buf = backendAllocator->get<cl_mem>(*tensor, *device);
      clSetKernelArg(kernel, i, sizeof(cl_mem), buf);
    } else {
      clSetKernelArg(kernel, i, cmd.args[i].size, cmd.args[i].arg);
    }
  }

  // todo check errors.
  clEnqueueNDRangeKernel(queue.getNativeQueue(), kernel, cmd.workDim,
                         nullptr, // global offset
                         cmd.globalSize, cmd.localSize,
                         0,       // num events in wait list
                         nullptr, // event list
                         nullptr  // event
  );
}
}
