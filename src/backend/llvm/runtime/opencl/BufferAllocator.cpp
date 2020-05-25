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

#include "BufferAllocator.h"

#include <athena/utils/error/FatalError.h>

namespace athena::backend::llvm {
void BufferAllocator::allocate(MemoryRecord record) {
  if (mBuffers.count(record))
    return; // no double allocations are allowed

  // todo re-use released buffers for new allocations.

  cl_int errCode;

  // fixme error checking
  cl_mem buffer = clCreateBuffer(mContext, CL_MEM_READ_WRITE,
                                 record.allocationSize, nullptr, &errCode);

  if (errCode == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
    freeMemory(record);

    buffer = clCreateBuffer(mContext, CL_MEM_READ_WRITE, record.allocationSize,
                            nullptr, &errCode);
  }

  if (errCode != CL_SUCCESS) {
    new utils::FatalError(utils::ATH_FATAL_OTHER,
                          "Failed to allocate OpenCL buffer!");
  }

  mBuffers[record] = buffer;
  mTags[record] = 1;
}
void BufferAllocator::deallocate(MemoryRecord record) {
  if (mLockedAllocations.count(record)) {
    new utils::FatalError(
        utils::ATH_FATAL_OTHER,
        "Attempt to deallocate locked buffer: ", record.virtualAddress);
  }

  if (!mBuffers.count(record)) {
    new utils::FatalError(utils::ATH_FATAL_OTHER,
                          "Double free of vaddr: ", record.virtualAddress);
  }

  if (mReleasedAllocations.count(record)) {
    mReleasedAllocations.erase(record);
  }

  clReleaseMemObject(mBuffers[record]);
  mBuffers.erase(record);
  mTags[record] = 0;
}
void athena::backend::llvm::BufferAllocator::lock(MemoryRecord record) {
  mLockedAllocations.insert(record);
}
void athena::backend::llvm::BufferAllocator::release(MemoryRecord record) {
  mLockedAllocations.erase(record);
  mReleasedAllocations.insert(record);
}
void* athena::backend::llvm::BufferAllocator::getPtr(
    llvm::MemoryRecord record) {
  if (mBuffers.count(record)) {
    return &mBuffers[record];
  }
  return nullptr;
}
void BufferAllocator::freeMemory(MemoryRecord record) {
  size_t freedMem = 0;
  while (freedMem < record.allocationSize) {
    if (mReleasedAllocations.size() == 0)
      new utils::FatalError(utils::ATH_FATAL_OTHER, "Out of memory!");
    MemoryRecord alloc = *mReleasedAllocations.begin();
    freedMem += alloc.allocationSize;
    mCallback(alloc, *this);
    clReleaseMemObject(mBuffers[alloc]);
    mBuffers.erase(alloc);
    mReleasedAllocations.erase(alloc);
  }
}
bool BufferAllocator::isAllocated(const MemoryRecord& record) const {
  return mBuffers.count(record) > 0;
}
size_t BufferAllocator::getTag(MemoryRecord record) { return mTags[record]; }
void BufferAllocator::setTag(MemoryRecord record, size_t tag) {
  mTags[record] = tag;
}
} // namespace athena::backend::llvm