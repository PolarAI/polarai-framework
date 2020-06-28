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

#include "HIPContext.h"
#include "HIPDevice.h"
#include "utils.hpp"

#include <hip/hip_runtime.h>

namespace athena::backend::llvm {
HIPContext::HIPContext() {
  hipError_t err = hipInit(0);

  if (err != hipSuccess) {
    std::terminate(); // fixme handle errors
  }

  int deviceCount;
  err = hipGetDeviceCount(&deviceCount);
  if (err != hipSuccess) {
    std::cerr << "Error while hipGetDeviceCount: " << decodeHipError(err)
              << "\n";
    std::terminate();
  }

  mDevices.reserve(deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    hipDevice_t device;
    err = hipDeviceGet(&device, i);
    if (err != hipSuccess) {
      continue;
    }
    mDevices.push_back(std::make_shared<HIPDevice>(device));
  }
}
std::vector<std::shared_ptr<Device>>& HIPContext::getDevices() {
  return mDevices;
}
} // namespace athena::backend::llvm
