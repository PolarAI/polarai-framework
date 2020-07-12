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

#ifndef ATHENA_OPENCLQUEUE_H
#define ATHENA_OPENCLQUEUE_H

#include <athena/backend/llvm/runtime/Queue.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace athena::backend::llvm {
class OpenClQueue : public Queue {
public:
  OpenClQueue(cl_context ctx, cl_device_id device) {
    // todo migrate to clCreateCommandQueueWithProperties
    // fixme handle errors.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    mQueue = clCreateCommandQueue(ctx, device, 0, nullptr);
#pragma GCC diagnistic pop
  }

  void wait() override { clFlush(mQueue); }

  cl_command_queue getNativeQueue() { return mQueue; }

private:
  cl_command_queue mQueue;
};
} // namespace athena::backend::llvm

#endif // ATHENA_OPENCLQUEUE_H
