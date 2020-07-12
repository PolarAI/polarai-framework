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

#pragma once

#include <CL/sycl.hpp>

#include "../../utils/utils.h"
#include "../SYCLDevice.h"
#include "../SYCLEvent.h"
#include "common.hpp"

#include <athena/backend/llvm/runtime/LaunchCommand.h>

template <int AllocT, typename T> class SigmoidKernel {
public:
  using ReadAcc = read_accessor_t<AllocT, T, 1>;
  using WriteAcc = discard_write_accessor_t<AllocT, T, 1>;

  SigmoidKernel(ReadAcc in, WriteAcc out) : in(in), out(out) {}

  void operator()(cl::sycl::id<1> id) {
    T eps = 1e-5;
    T res = 1 / (1 + cl::sycl::exp(-in[id]));
    if (cl::sycl::fabs(res-1) < eps) {
      res = 1 - eps;
    } else if (cl::sycl::fabs(res) < eps) {
      res = eps;
    }
    out[id] = res;
  }

private:
  ReadAcc in;
  WriteAcc out;
};

template <int AllocT, typename T> class SigmoidKernelWrapper;

template <typename T> class SigmoidKernelWrapper<AllocatorType::buffer, T> {
public:
  auto operator()(athena::backend::llvm::SYCLDevice* device,
                  athena::backend::llvm::BackendAllocator& allocator,
                  LaunchCommand& cmd, athena::backend::llvm::Event* evt)
      -> athena::backend::llvm::Event* {
    using namespace athena::backend::llvm;
    using namespace cl::sycl;

    auto aTensor = static_cast<TensorInfo*>(cmd.args[0].arg);
    MemoryRecord aRecord = tensorInfoToRecord(aTensor);

    auto cTensor = static_cast<TensorInfo*>(cmd.args[1].arg);
    MemoryRecord cRecord = tensorInfoToRecord(cTensor);

    auto aBuf = allocator.get<buffer<char, 1>>(aRecord, *device);
    auto cBuf = allocator.get<buffer<char, 1>>(cRecord, *device);

    buffer<T, 1> aTBuf =
        aBuf->reinterpret<T>(range<1>(aRecord.allocationSize / sizeof(T)));
    buffer<T, 1> cTBuf =
        cBuf->reinterpret<T>(range<1>(cRecord.allocationSize / sizeof(T)));

    auto q = device->getQueue().getNativeQueue();

    auto outEvt = q.submit([&](handler& cgh) {
      auto aAcc = aTBuf.template get_access<access::mode::read>(cgh);
      auto cAcc = cTBuf.template get_access<access::mode::discard_write>(cgh);
      SigmoidKernel<AllocatorType::buffer, T> kernel(aAcc, cAcc);

      cgh.parallel_for(cBuf->get_range(), kernel);
    });

    return new SYCLEvent(device, outEvt);
  }
};

template <typename T> class SigmoidKernelWrapper<AllocatorType::usm, T> {
public:
  auto operator()(athena::backend::llvm::SYCLDevice* device,
                  athena::backend::llvm::BackendAllocator& allocator,
                  LaunchCommand& cmd, athena::backend::llvm::Event* evt)
      -> athena::backend::llvm::Event* {
    using namespace athena::backend::llvm;
    using namespace cl::sycl;

    auto aTensor = static_cast<TensorInfo*>(cmd.args[0].arg);
    MemoryRecord aRecord = tensorInfoToRecord(aTensor);

    auto cTensor = static_cast<TensorInfo*>(cmd.args[1].arg);
    MemoryRecord cRecord = tensorInfoToRecord(cTensor);

    auto aBuf = allocator.get<T>(aRecord, *device);
    auto cBuf = allocator.get<T>(cRecord, *device);

    auto q = device->getQueue().getNativeQueue();

    auto outEvt = q.submit([&](handler& cgh) {
      auto aAcc = read_accessor_t<AllocatorType::usm, T, 1>(
          aBuf, range<1>(aRecord.allocationSize / sizeof(T)));
      auto cAcc = discard_write_accessor_t<AllocatorType::usm, T, 1>(
          cBuf, range<1>(cRecord.allocationSize / sizeof(T)));
      SigmoidKernel<AllocatorType::usm, T> kernel(aAcc, cAcc);

      cgh.parallel_for(range<1>(cRecord.allocationSize / sizeof(T)), kernel);
    });

    return new SYCLEvent(device, outEvt);
  }
};
