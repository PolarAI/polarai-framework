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

#include <athena/backend/llvm/runtime/Queue.h>

#include <CL/sycl.hpp>

namespace athena::backend::llvm {
class SYCLQueue : public Queue {
public:
  // todo add async exception handler
  SYCLQueue(const cl::sycl::device& device) : mQueue(device) {}

  void wait() override { mQueue.wait(); }

  cl::sycl::queue getNativeQueue() { return mQueue; }

private:
  cl::sycl::queue mQueue;
};
} // namespace athena::backend::llvm
