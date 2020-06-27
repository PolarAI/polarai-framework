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

namespace athena::backend::llvm {
CudaDevice::CudaDevice(CUdevice device) : mDevice(device) {
  char name[100];
  cuDeviceGetName(name, 100, mDevice);
  mDeviceName = std::string(name);
}

std::string CudaDevice::getDeviceName() const {
  return mDeviceName;
}
}
