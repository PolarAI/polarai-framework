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

template <int AllocT, typename T> class LogLossKernel {
public:
  using ReadAcc = read_accessor_t<AllocT, T, 1>;
  using WriteAcc = discard_write_accessor_t<AllocT, T, 1>;

  LogLossKernel(ReadAcc predicted, ReadAcc groundTruth, WriteAcc out)
      : predicted(predicted), groundTruth(groundTruth), out(out) {}

  void operator()(cl::sycl::id<1> id) {
    T eps = 1e-5;
    out[id] = -groundTruth[id] * cl::sycl::log(predicted[id] + eps) -
              (1 - groundTruth[id]) * cl::sycl::log(1 - predicted[id] + eps);
  }

private:
  ReadAcc predicted, groundTruth;
  WriteAcc out;
};

template <int AllocT, typename T> class LogLossWrapper;

template <typename T> class LogLossWrapper<AllocatorType::buffer, T> {
public:
  auto operator()(athena::backend::llvm::SYCLDevice* device,
                  athena::backend::llvm::BackendAllocator& allocator,
                  LaunchCommand& cmd, athena::backend::llvm::Event* evt)
      -> athena::backend::llvm::Event* {
    using namespace athena::backend::llvm;
    using namespace cl::sycl;

    auto predicted = static_cast<TensorInfo*>(cmd.args[0].arg);
    MemoryRecord predictedRecord = tensorInfoToRecord(predicted);

    auto groundTruth = static_cast<TensorInfo*>(cmd.args[1].arg);
    MemoryRecord groundTruthRecord = tensorInfoToRecord(groundTruth);

    auto out = static_cast<TensorInfo*>(cmd.args[2].arg);
    MemoryRecord outRecord = tensorInfoToRecord(out);

    auto predBufRaw = allocator.get<buffer<char, 1>>(predictedRecord, *device);
    auto truthBufRaw =
        allocator.get<buffer<char, 1>>(groundTruthRecord, *device);
    auto outBufRaw = allocator.get<buffer<char, 1>>(outRecord, *device);

    auto predBuf = predBufRaw->reinterpret<T>(
        range<1>(predictedRecord.allocationSize / sizeof(T)));
    auto truthBuf = truthBufRaw->reinterpret<T>(
        range<1>(groundTruthRecord.allocationSize / sizeof(T)));
    auto outBuf = outBufRaw->reinterpret<T>(
        range<1>(outRecord.allocationSize / sizeof(T)));

    auto q = device->getQueue().getNativeQueue();

    auto outEvt = q.submit([&](handler& cgh) {
      auto aAcc = predBuf.template get_access<access::mode::read>(cgh);
      auto bAcc = truthBuf.template get_access<access::mode::read>(cgh);
      auto cAcc = outBuf.template get_access<access::mode::discard_write>(cgh);
      LogLossKernel<AllocatorType::buffer, T> kernel(aAcc, bAcc, cAcc);

      cgh.parallel_for(range<1>(outRecord.allocationSize / sizeof(T)), kernel);
    });

    return new SYCLEvent(device, outEvt);
  }
};
template <typename T> class LogLossWrapper<AllocatorType::usm, T> {
public:
  auto operator()(athena::backend::llvm::SYCLDevice* device,
                  athena::backend::llvm::BackendAllocator& allocator,
                  LaunchCommand& cmd, athena::backend::llvm::Event* evt)
      -> athena::backend::llvm::Event* {
    using namespace athena::backend::llvm;
    using namespace cl::sycl;

    auto predicted = static_cast<TensorInfo*>(cmd.args[0].arg);
    MemoryRecord predictedRecord = tensorInfoToRecord(predicted);

    auto groundTruth = static_cast<TensorInfo*>(cmd.args[1].arg);
    MemoryRecord groundTruthRecord = tensorInfoToRecord(groundTruth);

    auto out = static_cast<TensorInfo*>(cmd.args[2].arg);
    MemoryRecord outRecord = tensorInfoToRecord(out);

    auto predBuf = allocator.get<T>(predictedRecord, *device);
    auto truthBuf = allocator.get<T>(groundTruthRecord, *device);
    auto outBuf = allocator.get<T>(outRecord, *device);

    auto q = device->getQueue().getNativeQueue();

    auto outEvt = q.submit([&](handler& cgh) {
      auto aAcc = read_accessor_t<AllocatorType::usm, T, 1>(
          predBuf, range<1>(predictedRecord.allocationSize / sizeof(T)));
      auto bAcc = read_accessor_t<AllocatorType::usm, T, 1>(
          truthBuf, range<1>(groundTruthRecord.allocationSize / sizeof(T)));
      auto cAcc = discard_write_accessor_t<AllocatorType::usm, T, 1>(
          outBuf, range<1>(outRecord.allocationSize / sizeof(T)));
      LogLossKernel<AllocatorType::usm, T> kernel(aAcc, bAcc, cAcc);

      cgh.parallel_for(range<1>(outRecord.allocationSize / sizeof(T)), kernel);
    });

    return new SYCLEvent(device, outEvt);
  }
};
