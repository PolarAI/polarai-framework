//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
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

#include <polarai/backend/generic/AllocatorLayerBase.hpp>
#include <polarai/backend/generic/MemoryRecord.hpp>
#include <polarai/utils/error/FatalError.hpp>

#include <unordered_map>
#include <unordered_set>

namespace polarai::backend::generic {
class TrivialAllocator : public AllocatorLayerBase {
private:
  MemoryOffloadCallbackT mOffloadCallback;
  std::unordered_map<MemoryRecord, void*> mMemMap;
  std::unordered_set<MemoryRecord> mLockedAllocations;
  std::unordered_set<MemoryRecord> mReleasedAllocations;
  std::unordered_map<MemoryRecord, int> mTags;

  void freeMemory(MemoryRecord record) {
    size_t freedMem = 0;
    while (freedMem < record.allocationSize) {
      if (mReleasedAllocations.size() == 0)
        new utils::FatalError(utils::ATH_FATAL_OTHER, "Out of memory!");
      MemoryRecord alloc = *mReleasedAllocations.begin();
      freedMem += alloc.allocationSize;
      mOffloadCallback(alloc, *this);
      delete[] static_cast<unsigned char*>(mMemMap[alloc]);
      mMemMap.erase(alloc);
      mReleasedAllocations.erase(alloc);
    }
  }

public:
  ~TrivialAllocator() override = default;

  void registerMemoryOffloadCallback(MemoryOffloadCallbackT function) override {
  }
  void allocate(MemoryRecord record) override {
    if (mMemMap.count(record))
      return; // no double allocations are allowed

    void* mem = new unsigned char[record.allocationSize];
    if (mem == nullptr) {
      freeMemory(record);
      mem = new unsigned char[record.allocationSize];
    }
    if (mem == nullptr)
      new utils::FatalError(utils::ATH_FATAL_OTHER,
                            "Failed to allocate RAM memory!");
    mMemMap[record] = mem;
    mTags[record] = 1;
  }
  void deallocate(MemoryRecord record) override {
    if (mLockedAllocations.count(record)) {
      new utils::FatalError(utils::ATH_BAD_ACCESS, "Double free on vaddr ",
                            record.virtualAddress);
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

  void* getPtr(MemoryRecord record) override { return mMemMap[record]; }

  bool isAllocated(const MemoryRecord& record) const override {
    return mMemMap.count(record) > 0;
  }

  size_t getTag(MemoryRecord record) override { return mTags[record]; }

  void setTag(MemoryRecord record, size_t tag) override { mTags[record] = tag; }
};
} // namespace polarai::backend::llvm
