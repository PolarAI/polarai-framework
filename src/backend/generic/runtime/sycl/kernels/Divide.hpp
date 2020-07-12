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

template <int AllocT, typename T> class DivideKernel {
public:
  using ReadAcc = read_accessor_t<AllocT, T, 1>;
  using WriteAcc = discard_write_accessor_t<AllocT, T, 1>;

  DivideKernel(ReadAcc numerator, ReadAcc denominator, WriteAcc out)
      : numerator(numerator), denominator(denominator), out(out) {}

  void operator()(cl::sycl::id<1> id) {
    out[id] = numerator[id] / denominator[id];
  }

private:
  ReadAcc numerator, denominator;
  WriteAcc out;
};

template <int AllocT, typename T> class DivideKernelWrapper;

template <typename T> class DivideKernelWrapper<AllocatorType::buffer, T> {
public:
  auto operator()(athena::backend::llvm::SYCLDevice* device,
                  athena::backend::llvm::BackendAllocator& allocator,
                  LaunchCommand& cmd, athena::backend::llvm::Event* evt)
      -> athena::backend::llvm::Event* {
    using namespace athena::backend::llvm;
    using namespace cl::sycl;

    auto numerator = static_cast<TensorInfo*>(cmd.args[0].arg);
    MemoryRecord numeratorRecord = tensorInfoToRecord(numerator);

    auto denominator = static_cast<TensorInfo*>(cmd.args[1].arg);
    MemoryRecord denominatorRecord = tensorInfoToRecord(denominator);

    auto out = static_cast<TensorInfo*>(cmd.args[2].arg);
    MemoryRecord outRecord = tensorInfoToRecord(out);

    auto numBuf = allocator.get<buffer<char, 1>>(numeratorRecord, *device);
    auto denBuf = allocator.get<buffer<char, 1>>(denominatorRecord, *device);
    auto outBuf = allocator.get<buffer<char, 1>>(outRecord, *device);

    uint64_t totalSize = tensorSize(numerator);

    buffer<T, 1> numTBuf = numBuf->reinterpret<T>(range<1>(totalSize));
    buffer<T, 1> denTBuf = denBuf->reinterpret<T>(range<1>(totalSize));
    buffer<T, 1> outTBuf = outBuf->reinterpret<T>(range<1>(totalSize));

    auto q = device->getQueue().getNativeQueue();

    auto outEvt = q.submit([&](handler& cgh) {
      auto aAcc = numTBuf.template get_access<access::mode::read>(cgh);
      auto bAcc = denTBuf.template get_access<access::mode::read>(cgh);
      auto cAcc = outTBuf.template get_access<access::mode::discard_write>(cgh);
      DivideKernel<AllocatorType::buffer, T> kernel(aAcc, bAcc, cAcc);

      cgh.parallel_for(range<1>(totalSize), kernel);
    });

    return new SYCLEvent(device, outEvt);
  }
};

template <typename T> class DivideKernelWrapper<AllocatorType::usm, T> {
public:
  auto operator()(athena::backend::llvm::SYCLDevice* device,
                  athena::backend::llvm::BackendAllocator& allocator,
                  LaunchCommand& cmd, athena::backend::llvm::Event* evt)
      -> athena::backend::llvm::Event* {
    using namespace athena::backend::llvm;
    using namespace cl::sycl;

    auto numerator = static_cast<TensorInfo*>(cmd.args[0].arg);
    MemoryRecord numeratorRecord = tensorInfoToRecord(numerator);

    auto denominator = static_cast<TensorInfo*>(cmd.args[1].arg);
    MemoryRecord denominatorRecord = tensorInfoToRecord(denominator);

    auto out = static_cast<TensorInfo*>(cmd.args[2].arg);
    MemoryRecord outRecord = tensorInfoToRecord(out);

    auto numBuf = allocator.get<T>(numeratorRecord, *device);
    auto denBuf = allocator.get<T>(denominatorRecord, *device);
    auto outBuf = allocator.get<T>(outRecord, *device);

    auto q = device->getQueue().getNativeQueue();

    auto outEvt = q.submit([&](handler& cgh) {
      auto aAcc = read_accessor_t<AllocatorType::usm, T, 1>(
          numBuf, range<1>(numeratorRecord.allocationSize / sizeof(T)));
      auto bAcc = read_accessor_t<AllocatorType::usm, T, 1>(
          denBuf, range<1>(denominatorRecord.allocationSize / sizeof(T)));
      auto cAcc = discard_write_accessor_t<AllocatorType::usm, T, 1>(
          outBuf, range<1>(outRecord.allocationSize / sizeof(T)));
      DivideKernel<AllocatorType::usm, T> kernel(aAcc, bAcc, cAcc);

      cgh.parallel_for(range<1>(outRecord.allocationSize / sizeof(T)), kernel);
    });

    return new SYCLEvent(device, outEvt);
  }
};
