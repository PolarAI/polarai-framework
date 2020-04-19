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

#ifndef ATHENA_BUFFERALLOCATOR_H
#define ATHENA_BUFFERALLOCATOR_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <athena/backend/llvm/AllocatorLayerBase.h>
#include <athena/backend/llvm/MemoryRecord.h>

#include <unordered_map>
#include <unordered_set>

namespace athena::backend::llvm {
class BufferAllocator : public AllocatorLayerBase {
public:
  explicit BufferAllocator(cl_context ctx) : mContext(ctx) {}
  ~BufferAllocator() override = default;
  // We don't support offloading to OpenCL buffers, so, no implementation.
  void registerMemoryOffloadCallback(MemoryOffloadCallbackT t) override {}
  void allocate(MemoryRecord record) override;
  void deallocate(MemoryRecord record) override;
  void lock(MemoryRecord record) override;
  void release(MemoryRecord record) override;
  void* getPtr(MemoryRecord record) override;

private:
  void freeMemory(MemoryRecord record);

  cl_context mContext;
  MemoryOffloadCallbackT mCallback;
  std::unordered_map<MemoryRecord, cl_mem> mBuffers;
  std::unordered_set<MemoryRecord> mLockedAllocations;
  std::unordered_set<MemoryRecord> mReleasedAllocations;
};
}

#endif // ATHENA_BUFFERALLOCATOR_H
