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
}
