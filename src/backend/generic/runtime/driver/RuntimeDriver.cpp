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

#include "RuntimeDriver.hpp"
#include "DynamicLibrary.hpp"
#include "config.h"

#include <polarai/backend/generic/runtime/Device.hpp>

#include <iostream>
#include <string>

namespace polarai::backend::generic {
RuntimeDriver::RuntimeDriver(bool debugOutput) {
  auto libraries = getListOfLibraries();

  for (const auto& lib : libraries) {
    auto dynLib = DynamicLibrary::create(lib);

    if (!dynLib->isValid()) {
      if (debugOutput) {
        std::clog << "Failed to load " << lib << '\n';
        std::clog << dynLib->getLastError();
      }
      continue;
    } else if (debugOutput) {
      std::clog << "Successfully loaded " << lib << '\n';
    }

    void* initCtxPtr = dynLib->lookup("initContext");
    auto initCtxFunc = reinterpret_cast<Context* (*)()>(initCtxPtr);
    void* closeCtxPtr = dynLib->lookup("closeContext");
    auto closeCtxFunc = reinterpret_cast<void (*)(Context*)>(closeCtxPtr);

    Context* ctx = initCtxFunc();
    mContexts.emplace_back(ctx,
                           [closeCtxFunc](Context* ctx) { closeCtxFunc(ctx); });
    auto& newDevs = ctx->getDevices();
    mDevices.insert(mDevices.end(), newDevs.begin(), newDevs.end());

    mLibs.push_back(std::move(dynLib));
  }
}

} // namespace polarai::backend::generic
