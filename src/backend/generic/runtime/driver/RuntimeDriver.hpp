//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#pragma once

#include "DynamicLibrary.hpp"

#include <polarai/backend/generic/runtime/Device.hpp>
#include <polarai/backend/generic/runtime/Context.hpp>
#include <polar_generic_be_driver_export.h>

namespace polarai::backend::generic {
class POLAR_GENERIC_BE_DRIVER_EXPORT RuntimeDriver {
public:
  RuntimeDriver(bool enableDebugOutput);
  auto getDeviceList() -> std::vector<std::shared_ptr<Device>>& {
    return mDevices;
  };

private:
  std::vector<std::unique_ptr<DynamicLibrary>> mLibs;
  std::vector<std::shared_ptr<Device>> mDevices;
  std::vector<std::shared_ptr<Context>> mContexts;
};
} // namespace polarai::backend::llvm
