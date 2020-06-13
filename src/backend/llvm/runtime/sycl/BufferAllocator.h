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

#include <athena/backend/llvm/AllocatorLayerBase.h>
#include <athena/backend/llvm/MemoryRecord.h>

#include <CL/sycl.hpp>

#include <unordered_map>
#include <unordered_set>

namespace athena::backend::llvm {
class BufferAllocator : public AllocatorLayerBase {
public:
  explicit BufferAllocator(cl::sycl::context ctx) : mContext(std::move(ctx)) {}
  ~BufferAllocator() override = default;
  // We don't support offloading to OpenCL buffers, so, no implementation.
  void registerMemoryOffloadCallback(MemoryOffloadCallbackT t) override {}
  void allocate(MemoryRecord record) override;
  void deallocate(MemoryRecord record) override;
  void lock(MemoryRecord record) override;
  void release(MemoryRecord record) override;
  void* getPtr(MemoryRecord record) override;
  bool isAllocated(const MemoryRecord& record) const override;
  size_t getTag(MemoryRecord record) override;
  void setTag(MemoryRecord record, size_t tag) override;

private:
  void freeMemory(MemoryRecord record);

  cl::sycl::context mContext;
  MemoryOffloadCallbackT mCallback;
  std::unordered_map<MemoryRecord, cl::sycl::buffer<char, 1>> mBuffers;
  std::unordered_set<MemoryRecord> mLockedAllocations;
  std::unordered_set<MemoryRecord> mReleasedAllocations;
  std::unordered_map<MemoryRecord, int> mTags;
};
} // namespace athena::backend::llvm
