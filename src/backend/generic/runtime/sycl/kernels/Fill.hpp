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

template <int AllocT, typename T> class FillWrapper;

template <typename T> class FillWrapper<AllocatorType::buffer, T> {
public:
  auto operator()(athena::backend::llvm::SYCLDevice* device,
                  athena::backend::llvm::BackendAllocator& alloc,
                  LaunchCommand& cmd, athena::backend::llvm::Event* evt)
      -> athena::backend::llvm::Event* {
    using namespace cl::sycl;
    using namespace athena::backend::llvm;
    auto q = device->getQueue().getNativeQueue();
    // todo assert on arguments

    auto tensor = static_cast<TensorInfo*>(cmd.args[1].arg);
    MemoryRecord record = tensorInfoToRecord(tensor);

    auto* buf = alloc.get<buffer<char, 1>>(record, *device);
    buffer<float, 1> reintBuf =
        buf->reinterpret<float, 1>(range<1>(buf->get_count() / 4));
    T pattern = *static_cast<T*>(cmd.args[0].arg);
    auto outEvt = q.submit([&reintBuf, pattern](handler& cgh) {
      auto acc = reintBuf.template get_access<access::mode::discard_write>(cgh);
      cgh.template fill(acc, pattern);
    });

    return new SYCLEvent(device, outEvt);
  }
};

template <typename T> class FillWrapper<AllocatorType::usm, T> {
public:
  auto operator()(athena::backend::llvm::SYCLDevice* device,
                  athena::backend::llvm::BackendAllocator& alloc,
                  LaunchCommand& cmd, athena::backend::llvm::Event* evt)
      -> athena::backend::llvm::Event* {
    using namespace cl::sycl;
    using namespace athena::backend::llvm;
    auto q = device->getQueue().getNativeQueue();
    // todo assert on arguments

    auto tensor = static_cast<TensorInfo*>(cmd.args[1].arg);
    MemoryRecord record = tensorInfoToRecord(tensor);

    T* buf = alloc.get<T>(record, *device);
    T pattern = *static_cast<T*>(cmd.args[0].arg);
    auto outEvt = q.submit([&](handler& cgh) {
      cgh.parallel_for<class fill>(range<1>(record.allocationSize / sizeof(T)),
                       [=](cl::sycl::id<1> cur) {
        buf[cur.get(0)] = pattern;
      });
      // FIXME use shortcuts once they are implemented
      // cgh.template fill<T>(buf, pattern, record.allocationSize / sizeof(T));
    });

    return new SYCLEvent(device, outEvt);
  }
};
