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

#include "SYCLContext.h"
#include "SYCLDevice.h"

#include <CL/sycl.hpp>

using namespace cl::sycl;

namespace athena::backend::llvm {
SYCLContext::SYCLContext() {
  auto allDevices = device::get_devices(cl::sycl::info::device_type::all);

  #ifdef USES_COMPUTECPP
  if (allDevices.size() > 1) {
    allDevices.erase(allDevices.begin()); // remove host device
  }
  #endif

  mDevices.reserve(allDevices.size());

  for (auto& dev : allDevices) {
    mDevices.push_back(std::make_shared<SYCLDevice>(dev));
  }
}
std::vector<std::shared_ptr<Device>>& SYCLContext::getDevices() {
  return mDevices;
}
}

