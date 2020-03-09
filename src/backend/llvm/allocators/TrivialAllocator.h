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

#ifndef ATHENA_TRIVIALALLOCATOR_H
#define ATHENA_TRIVIALALLOCATOR_H

#include <athena/backend/llvm/AllocatorLayerBase.h>
#include <athena/backend/llvm/MemoryRecord.h>
#include <athena/core/FatalError.h>

#include <unordered_map>
#include <unordered_set>

namespace athena::backend::llvm {
class TrivialAllocator : public AllocatorLayerBase {
private:
  memoryOffloadCallbackT mOffloadCallback;
  std::unordered_map<MemoryRecord, void*> mMemMap;
  std::unordered_set<MemoryRecord> mLockedAllocations;
  std::unordered_set<MemoryRecord> mReleasedAllocations;

  void freeMemory(MemoryRecord record) {
    size_t freedMem = 0;
    while (freedMem < record.allocationSize) {
      if (mReleasedAllocations.size() == 0)
        new core::FatalError(core::ATH_FATAL_OTHER, "Out of memory!");
      MemoryRecord alloc = *mReleasedAllocations.begin();
      freedMem += alloc.allocationSize;
      mOffloadCallback(alloc);
      delete[] static_cast<unsigned char*>(mMemMap[alloc]);
      mMemMap.erase(alloc);
      mReleasedAllocations.erase(alloc);
    }
  }

public:
  void registerMemoryOffloadCallback(
      std::function<void(MemoryRecord)> function) override {}
  void allocate(MemoryRecord record) override {
    if (mMemMap.count(record))
      return; // no double allocations are allowed

    void* mem = new unsigned char[record.allocationSize];
    if (mem == nullptr) {
      freeMemory(record);
      mem = new unsigned char[record.allocationSize];
    }
    if (mem == nullptr)
      new core::FatalError(core::ATH_FATAL_OTHER,
                           "Failed to allocate RAM memory!");
    mMemMap[record] = mem;
  }
  void deallocate(MemoryRecord record) override {
    if (mLockedAllocations.count(record)) {
      new core::FatalError(core::ATH_BAD_ACCESS, "Double free on vaddr ",
                           record.virtualAddress);
    }

    delete[] reinterpret_cast<unsigned char*>(mMemMap[record]);

    if (mReleasedAllocations.count(record)) {
      mReleasedAllocations.erase(record);
    }
  }
  void lock(MemoryRecord record) override { mLockedAllocations.insert(record); }
  void release(MemoryRecord record) override {
    mLockedAllocations.erase(record);
    mReleasedAllocations.insert(record);
  }

  void* getPtr(MemoryRecord record) override { return mMemMap[record]; }
};
} // namespace athena::backend::llvm

#endif // ATHENA_TRIVIALALLOCATOR_H
