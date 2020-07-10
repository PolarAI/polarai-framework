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
#include <athena/utils/error/FatalError.h>

#include <unordered_map>
#include <unordered_set>

#import <Metal/Metal.h>

namespace athena::backend::llvm {
class MetalAllocator : public AllocatorLayerBase {
private:
  MemoryOffloadCallbackT mOffloadCallback;
  id<MTLDevice> mDevice;
  std::unordered_map<MemoryRecord, id<MTLBuffer>> mMemMap;
  std::unordered_set<MemoryRecord> mLockedAllocations;
  std::unordered_set<MemoryRecord> mReleasedAllocations;
  std::unordered_map<MemoryRecord, int> mTags;

  void freeMemory(MemoryRecord record) {
    size_t freedMem = 0;
    while (freedMem < record.allocationSize) {
      if (mReleasedAllocations.size() == 0)
        std::terminate();
      MemoryRecord alloc = *mReleasedAllocations.begin();
      freedMem += alloc.allocationSize;
      mOffloadCallback(alloc, *this);
      // todo free buffers
      mMemMap.erase(alloc);
      mReleasedAllocations.erase(alloc);
    }
  }

public:
  MetalAllocator(id<MTLDevice> device) : mDevice(device){};
  ~MetalAllocator() override = default;

  void registerMemoryOffloadCallback(MemoryOffloadCallbackT function) override {
  }
  void allocate(MemoryRecord record) override {
    if (mMemMap.count(record))
      return; // no double allocations are allowed

    auto mem = [mDevice newBufferWithLength:(record.allocationSize)
                                    options:(MTLResourceStorageModeShared)];

    mMemMap[record] = mem;
    mTags[record] = 1;
  }
  void deallocate(MemoryRecord record) override {
    if (mLockedAllocations.count(record)) {
      std::terminate();
    }

    delete[] reinterpret_cast<unsigned char*>(mMemMap[record]);

    if (mReleasedAllocations.count(record)) {
      mReleasedAllocations.erase(record);
    }
    mTags[record] = 0;
  }
  void lock(MemoryRecord record) override { mLockedAllocations.insert(record); }
  void release(MemoryRecord record) override {
    mLockedAllocations.erase(record);
    mReleasedAllocations.insert(record);
  }

  void* getPtr(MemoryRecord record) override { return &mMemMap[record]; }

  bool isAllocated(const MemoryRecord& record) const override {
    return mMemMap.count(record) > 0;
  }

  size_t getTag(MemoryRecord record) override { return mTags[record]; }

  void setTag(MemoryRecord record, size_t tag) override { mTags[record] = tag; }
};
} // namespace athena::backend::llvm
