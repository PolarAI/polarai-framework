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

#include "RuntimeDriver.h"
#include "DynamicLibrary.h"
#include "config.h"

#include <athena/backend/llvm/runtime/Device.h>


#include <iostream>
#include <string>

namespace athena::backend::llvm {
RuntimeDriver::RuntimeDriver() {
  auto libraries = getListOfLibraries();

  for (const auto& lib : libraries) {
    auto dynLib = DynamicLibrary::create(lib);

    void* initCtxPtr = dynLib->lookup("initContext");
    auto initCtxFunc = reinterpret_cast<Context* (*)()>(initCtxPtr);
    void* closeCtxPtr = dynLib->lookup("closeContext");
    auto closeCtxFunc = reinterpret_cast<void (*)(Context*)>(closeCtxPtr);

    Context* ctx = initCtxFunc();
    mContexts.emplace_back(ctx,
                           [closeCtxFunc](Context* ctx) { closeCtxFunc(ctx); });
    auto &newDevs = ctx->getDevices();
    mDevices.insert(mDevices.end(), newDevs.begin(), newDevs.end());

    mLibs.push_back(std::move(dynLib));
  }
}

} // namespace athena::backend::llvm
