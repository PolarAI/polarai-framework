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

#include <functional>

namespace athena::backend::llvm {

// Forward declaration
struct MemoryRecord;

class AllocatorLayerBase {
public:
  using memoryOffloadCallbackT = std::function<void(MemoryRecord)>;

  virtual ~AllocatorLayerBase() = default;

  virtual void registerMemoryOffloadCallback(memoryOffloadCallbackT) = 0;

  virtual void allocate(MemoryRecord record) = 0;
  virtual void deallocate(MemoryRecord record) = 0;
  virtual void lock(MemoryRecord record) = 0;
  virtual void release(MemoryRecord record) = 0;
  virtual void* getPtr(MemoryRecord record) = 0;
};
}
