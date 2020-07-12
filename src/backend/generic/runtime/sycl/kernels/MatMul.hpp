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

template <int AllocT, typename T, bool TranspLeft, bool TranspRight>
class MatMulKernel {
public:
  using ReadAcc = read_accessor_t<AllocT, T, 2>;
  using WriteAcc = discard_write_accessor_t<AllocT, T, 2>;

  MatMulKernel(ReadAcc a, ReadAcc b, WriteAcc c) : left(a), right(b), out(c) {}

  void operator()(cl::sycl::id<2> id) {
    size_t K = 0;
    if constexpr (TranspLeft) {
      K = left.get_range()[0];
    } else {
      K = left.get_range()[1];
    }
    T acc = 0;
    for (int k = 0; k < K; k++) {
      size_t leftRow = 0;
      size_t leftCol = 0;
      if constexpr (TranspLeft) {
        leftRow = k;
        leftCol = id[0];
      } else {
        leftRow = id[0];
        leftCol = k;
      }
      size_t rightRow = 0;
      size_t rightCol = 0;
      if constexpr (TranspRight) {
        rightRow = id[1];
        rightCol = k;
      } else {
        rightRow = k;
        rightCol = id[1];
      }
      acc += left[cl::sycl::id<2>{leftRow, leftCol}] *
            right[cl::sycl::id<2>{rightRow, rightCol}];
    }
    out[id] = acc;
  }

private:
  ReadAcc left, right;
  WriteAcc out;
};

template <int AllocT, typename T, bool TranspLeft, bool TranspRight>
class MatMulKernelWrapper;

template <typename T, bool TranspLeft, bool TranspRight>
class MatMulKernelWrapper<AllocatorType::buffer, T, TranspLeft, TranspRight> {
public:
  auto operator()(athena::backend::llvm::SYCLDevice* device,
                  athena::backend::llvm::BackendAllocator& allocator,
                  LaunchCommand& cmd, athena::backend::llvm::Event* evt)
      -> athena::backend::llvm::Event* {
    using namespace athena::backend::llvm;
    using namespace cl::sycl;

    auto aTensor = static_cast<TensorInfo*>(cmd.args[0].arg);
    MemoryRecord aRecord = tensorInfoToRecord(aTensor);

    auto bTensor = static_cast<TensorInfo*>(cmd.args[1].arg);
    MemoryRecord bRecord = tensorInfoToRecord(bTensor);

    auto cTensor = static_cast<TensorInfo*>(cmd.args[2].arg);
    MemoryRecord cRecord = tensorInfoToRecord(cTensor);

    auto aBuf = allocator.get<buffer<char, 1>>(aRecord, *device);
    auto bBuf = allocator.get<buffer<char, 1>>(bRecord, *device);
    auto cBuf = allocator.get<buffer<char, 1>>(cRecord, *device);

    buffer<T, 2> aTBuf =
        aBuf->reinterpret<T>(range<2>(aTensor->shape[0], aTensor->shape[1]));
    buffer<T, 2> bTBuf =
        bBuf->reinterpret<T>(range<2>(bTensor->shape[0], bTensor->shape[1]));
    buffer<T, 2> cTBuf =
        cBuf->reinterpret<T>(range<2>(cTensor->shape[0], cTensor->shape[1]));

    auto q = device->getQueue().getNativeQueue();

    auto outEvt = q.submit([&](handler& cgh) {
      auto aAcc = aTBuf.template get_access<access::mode::read>(cgh);
      auto bAcc = bTBuf.template get_access<access::mode::read>(cgh);
      auto cAcc = cTBuf.template get_access<access::mode::discard_write>(cgh);
      MatMulKernel<AllocatorType::buffer, T, TranspLeft, TranspRight> kernel(
          aAcc, bAcc, cAcc);

      cgh.parallel_for(cTBuf.get_range(), kernel);
    });

    return new SYCLEvent(device, outEvt);
  }
};

template <typename T, bool TranspLeft, bool TranspRight>
class MatMulKernelWrapper<AllocatorType::usm, T, TranspLeft, TranspRight> {
public:
  auto operator()(athena::backend::llvm::SYCLDevice* device,
                  athena::backend::llvm::BackendAllocator& allocator,
                  LaunchCommand& cmd, athena::backend::llvm::Event* evt)
      -> athena::backend::llvm::Event* {
    using namespace athena::backend::llvm;
    using namespace cl::sycl;

    auto aTensor = static_cast<TensorInfo*>(cmd.args[0].arg);
    MemoryRecord aRecord = tensorInfoToRecord(aTensor);

    auto bTensor = static_cast<TensorInfo*>(cmd.args[1].arg);
    MemoryRecord bRecord = tensorInfoToRecord(bTensor);

    auto cTensor = static_cast<TensorInfo*>(cmd.args[2].arg);
    MemoryRecord cRecord = tensorInfoToRecord(cTensor);

    auto aBuf = allocator.get<T>(aRecord, *device);
    auto bBuf = allocator.get<T>(bRecord, *device);
    auto cBuf = allocator.get<T>(cRecord, *device);

    auto q = device->getQueue().getNativeQueue();

    auto outEvt = q.submit([&](handler& cgh) {
      auto aAcc = read_accessor_t<AllocatorType::usm, T, 2>(
          aBuf, range<2>(aTensor->shape[0], aTensor->shape[1]));
      auto bAcc = read_accessor_t<AllocatorType::usm, T, 2>(
          bBuf, range<2>(bTensor->shape[0], bTensor->shape[1]));
      auto cAcc = discard_write_accessor_t<AllocatorType::usm, T, 2>(
          cBuf, range<2>(cTensor->shape[0], cTensor->shape[1]));
      MatMulKernel<AllocatorType::usm, T, TranspLeft, TranspRight> kernel(
          aAcc, bAcc, cAcc);

      cgh.parallel_for(range<2>(cAcc.get_range()), kernel);
    });

    return new SYCLEvent(device, outEvt);
  }
};
