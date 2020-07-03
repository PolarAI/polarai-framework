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

#include "CudaDevice.h"
#include "../utils/utils.h"
#include "CudaEvent.h"
#include "utils.hpp"

#include <athena/backend/llvm/BackendAllocator.h>
#include <athena/backend/llvm/runtime/LaunchCommand.h>

#include <array>
#include <fstream>
#include <vector>

namespace athena::backend::llvm {
CudaDevice::CudaDevice(CUdevice device) : mDevice(device) {
  std::array<char, 100> name;
  check(cuDeviceGetName(name.data(), 100, mDevice));
  mDeviceName = std::string(name.data());

  check(cuCtxCreate(&mDeviceContext, 0, mDevice));
  check(cuStreamCreate(&mStream, 0));

  mAllocator = std::make_shared<CudaAllocator>();
}

std::string CudaDevice::getDeviceName() const { return mDeviceName; }

void CudaDevice::selectBinary(
    std::vector<std::shared_ptr<ProgramDesc>>& programs) {
  cuCtxSetCurrent(mDeviceContext);
  // todo real image selection logic
  auto prg = programs[0];

  std::array<char, 4096> jitErrorBuffer = {0};

  CUlinkState linkState;

  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void* jitOptionsVals[] = {jitErrorBuffer.data(),
                            reinterpret_cast<void*>(jitErrorBuffer.size())};
  check(cuLinkCreate(2, jitOptions, jitOptionsVals, &linkState));
  check(cuLinkAddData(linkState, CUjitInputType::CU_JIT_INPUT_PTX,
                      static_cast<void*>(prg->data.data()), prg->data.size(),
                      "kernels", 0, nullptr, nullptr));
}
void CudaDevice::consumeEvent(Event* evt) {
  evt->wait();
  delete evt;
}

Event* CudaDevice::launch(BackendAllocator& allocator, LaunchCommand& cmd,
                          Event* blockingEvent) {
  cuCtxSetCurrent(mDeviceContext);
  uint64_t zero = 0;
  std::vector<void*> args;

  for (int i = 0; i < cmd.argsCount; i++) {
    if (cmd.args[i].type == ArgDesc::TENSOR) {
      auto tensor = static_cast<TensorInfo*>(cmd.args[i].arg);
      auto record = tensorInfoToRecord(tensor);
      auto buf = allocator.get<CUdeviceptr>(record, *this);
      args.push_back(buf);
      args.push_back(buf);

      args.push_back(&zero); // offset

      for (int dim = 0; dim < tensor->dims; dim++) {
        args.push_back(&tensor->shape[dim]); // size
      }
      for (int dim = 0; dim < tensor->dims; dim++) {
        args.push_back(&zero); // stride
      }
    } else {
      args.push_back(cmd.args[i].arg);
    }
  }

  size_t gridX = 1, gridY = 1, gridZ = 1, blockX = 1, blockY = 1, blockZ = 1;

  gridX = cmd.globalSize[0];

  if (cmd.workDim > 1) {
    gridY = cmd.globalSize[1];
  }
  if (cmd.workDim > 2) {
    gridZ = cmd.globalSize[2];
  }

  CUfunction func;
  check(cuModuleGetFunction(&func, mMainModule, cmd.kernelName));

  check(cuLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY, blockZ, 0,
                       mStream, args.data(), nullptr));
  check(cuStreamSynchronize(mStream));
  CUevent evt;
  check(cuEventCreate(&evt, 0));
  check(cuEventRecord(evt, mStream));

  return new CudaEvent(this, evt);
}
CudaDevice::~CudaDevice() {
  cuModuleUnload(mMainModule);
  cuCtxDestroy(mDeviceContext);
}
} // namespace athena::backend::llvm
