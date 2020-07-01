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

#include "SYCLDevice.h"
#include "../utils/utils.h"
#include "SYCLEvent.h"

#include <athena/backend/llvm/BackendAllocator.h>
#include <athena/backend/llvm/runtime/LaunchCommand.h>
#include <athena/backend/llvm/runtime/TensorInfo.h>

#include "kernels/AddNaive.hpp"
#include "kernels/Copy.hpp"
#include "kernels/Divide.hpp"
#include "kernels/Fill.hpp"
#include "kernels/LogLoss.hpp"
#include "kernels/MatMul.hpp"
#include "kernels/Mul.hpp"
#include "kernels/MulConcat.hpp"
#include "kernels/Sigmoid.hpp"

using namespace cl::sycl;

namespace athena::backend::llvm {
auto SYCLDevice::launch(BackendAllocator& allocator, LaunchCommand& cmd,
                        Event* dependency) -> Event* {
  if (dependency) {
    // fixme there's no clear way in SYCL to launch kernel after the event is
    //  completed. But due to the barrier syncs this most likely will be a
    //  non-blocking call. In future runtimes will need a unified event system.
    dependency->wait();
  }
  if (mKernelMap.count(cmd.nativeKernelName)) {
    return mKernelMap.at(cmd.nativeKernelName)(this, allocator, cmd, dependency);
  }

  // todo implement interoperability kernel launch
  throw std::runtime_error("Not implemented");
  return nullptr;
}

void SYCLDevice::addModule(ProgramDesc) {}
void SYCLDevice::linkModules() {}

void SYCLDevice::consumeEvent(Event* evt) {
  evt->wait();
  delete evt;
}

void SYCLDevice::populateKernelMap() {
  if (mUsesUSM) {
    mKernelMap["ffill"] = FillWrapper<AllocatorType::usm, float>{};
    mKernelMap["fcopy"] = CopyWrapper<AllocatorType::usm, float>{};
    mKernelMap["fadd"] = AddKernelWrapper<AllocatorType::usm, float>{};
    mKernelMap["fdivide"] = DivideKernelWrapper<AllocatorType::usm, float>{};
    mKernelMap["flogloss"] = LogLossWrapper<AllocatorType::usm, float>{};
    mKernelMap["fmul"] = MulKernelWrapper<AllocatorType::usm, float>{};
    mKernelMap["fmulconcat"] =
        MulConcatKernelWrapper<AllocatorType::usm, float>{};
    mKernelMap["fsigmoid"] = SigmoidKernelWrapper<AllocatorType::usm, float>{};
    mKernelMap["fmatmul_f_f"] =
        MatMulKernelWrapper<AllocatorType::usm, float, false, false>{};
    mKernelMap["fmatmul_f_t"] =
        MatMulKernelWrapper<AllocatorType::usm, float, false, true>{};
    mKernelMap["fmatmul_t_f"] =
        MatMulKernelWrapper<AllocatorType::usm, float, true, false>{};
    mKernelMap["fmatmul_t_t"] =
        MatMulKernelWrapper<AllocatorType::usm, float, true, true>{};
  } else {
    mKernelMap["ffill"] = FillWrapper<AllocatorType::buffer, float>{};
    mKernelMap["fcopy"] = CopyWrapper<AllocatorType::buffer, float>{};
    mKernelMap["fadd"] = AddKernelWrapper<AllocatorType::buffer, float>{};
    mKernelMap["fdivide"] = DivideKernelWrapper<AllocatorType::buffer, float>{};
    mKernelMap["flogloss"] = LogLossWrapper<AllocatorType::buffer, float>{};
    mKernelMap["fmul"] = MulKernelWrapper<AllocatorType::buffer, float>{};
    mKernelMap["fmulconcat"] =
        MulConcatKernelWrapper<AllocatorType::buffer, float>{};
    mKernelMap["fsigmoid"] = SigmoidKernelWrapper<AllocatorType::buffer, float>{};
    mKernelMap["fmatmul_f_f"] =
        MatMulKernelWrapper<AllocatorType::buffer, float, false, false>{};
    mKernelMap["fmatmul_f_t"] =
        MatMulKernelWrapper<AllocatorType::buffer, float, false, true>{};
    mKernelMap["fmatmul_t_f"] =
        MatMulKernelWrapper<AllocatorType::buffer, float, true, false>{};
    mKernelMap["fmatmul_t_t"] =
        MatMulKernelWrapper<AllocatorType::buffer, float, true, true>{};

  }
}
} // namespace athena::backend::llvm
