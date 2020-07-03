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

#include "CudaContext.h"
#include "CudaDevice.h"
#include "utils.hpp"

#include <cuda.h>

namespace athena::backend::llvm {
CudaContext::CudaContext() {
  check(cuInit(0));

  int deviceCount;
  check(cuDeviceGetCount(&deviceCount));

  mDevices.reserve(deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    CUdevice device;
    check(cuDeviceGet(&device, i));
    mDevices.push_back(std::make_shared<CudaDevice>(device));
  }
}
std::vector<std::shared_ptr<Device>>& CudaContext::getDevices() {
  return mDevices;
}
}
