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

template <int AllocT, typename T> class CopyWrapper;

template <typename T> class CopyWrapper<AllocatorType::buffer, T> {
public:
  auto operator()(athena::backend::llvm::SYCLDevice* device,
                  athena::backend::llvm::BackendAllocator& alloc,
                  LaunchCommand& cmd, athena::backend::llvm::Event* evt)
      -> athena::backend::llvm::Event* {
    using namespace cl::sycl;
    using namespace athena::backend::llvm;
    auto q = device->getQueue().getNativeQueue();
    // todo assert on arguments

    auto srcTensor = static_cast<TensorInfo*>(cmd.args[0].arg);
    MemoryRecord srcRecord = tensorInfoToRecord(srcTensor);
    auto destTensor = static_cast<TensorInfo*>(cmd.args[1].arg);
    MemoryRecord destRecord = tensorInfoToRecord(destTensor);

    auto* bufSrc = alloc.get<buffer<char, 1>>(srcRecord, *device);
    auto* bufDst = alloc.get<buffer<char, 1>>(destRecord, *device);
    buffer<T, 1> src =
        bufSrc->template reinterpret<T, 1>(range<1>(bufSrc->get_count() / sizeof(T)));
    buffer<T, 1> dest =
        bufDst->template reinterpret<T, 1>(range<1>(bufDst->get_count() / sizeof(T)));
    auto outEvt = q.submit([&src, &dest](handler& cgh) {
      auto srcAcc = src.template get_access<access::mode::read>(cgh);
      auto dstAcc = dest.template get_access<access::mode::discard_write>(cgh);
      cgh.copy(srcAcc, dstAcc);
    });

    return new SYCLEvent(device, outEvt);
  }
};

template <typename T> class CopyWrapper<AllocatorType::usm, T> {
public:
  auto operator()(athena::backend::llvm::SYCLDevice* device,
                  athena::backend::llvm::BackendAllocator& alloc,
                  LaunchCommand& cmd, athena::backend::llvm::Event* evt)
      -> athena::backend::llvm::Event* {
    using namespace cl::sycl;
    using namespace athena::backend::llvm;
    auto q = device->getQueue().getNativeQueue();
    // todo assert on arguments

    auto srcTensor = static_cast<TensorInfo*>(cmd.args[0].arg);
    MemoryRecord srcRecord = tensorInfoToRecord(srcTensor);
    auto destTensor = static_cast<TensorInfo*>(cmd.args[1].arg);
    MemoryRecord destRecord = tensorInfoToRecord(destTensor);

    T* src = alloc.get<T>(srcRecord, *device);
    T* dest = alloc.get<T>(destRecord, *device);
    auto outEvt = q.memcpy(dest, src, srcRecord.allocationSize / sizeof(T));

    return new SYCLEvent(device, outEvt);
  }
};

