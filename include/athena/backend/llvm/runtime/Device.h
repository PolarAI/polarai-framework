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

#ifndef ATHENA_DEVICE_H
#define ATHENA_DEVICE_H

#include <athena/backend/llvm/AllocatorLayerBase.h>
#include <athena/backend/llvm/MemoryRecord.h>
#include <athena/backend/llvm/llvm_export.h>
#include <athena/backend/llvm/runtime/ProgramDesc.h>
#include <athena/backend/llvm/runtime/Queue.h>
#include <athena/core/tensor/internal/TensorInternal.h>

#include <cstddef>
#include <memory>

struct LaunchCommand;

namespace athena::backend::llvm {

class Device;
class Event;
class BackendAllocator;

extern "C" struct ATH_BACKEND_LLVM_EXPORT DeviceContainer {
  Device* devices;
  size_t count;
};

class ATH_BACKEND_LLVM_EXPORT Device {
public:
  Device() = default;

  enum class PartitionDomain { EQUALLY, BY_COUNT, NUMA };
  ///@{ \name Device information
  virtual std::string getDeviceName() const = 0;
  virtual bool isPartitionSupported(PartitionDomain domain) { return false; };
  virtual bool hasAllocator() { return false; };
  ///@}
  virtual DeviceContainer partition(PartitionDomain domain) = 0;
  virtual std::shared_ptr<AllocatorLayerBase> getAllocator() = 0;

  virtual bool operator==(const Device& device) const { return false; };
  bool operator!=(const Device& device) const { return !(*this == device); };

  virtual void copyToHost(const core::internal::TensorInternal& tensor,
                          void* dest) const = 0;
  virtual void copyToHost(MemoryRecord record, void* dest) const = 0;
  virtual void copyToDevice(const core::internal::TensorInternal& tensor,
                            void* src) const = 0;
  virtual void copyToDevice(MemoryRecord record, void* src) const = 0;

  virtual Event* launch(BackendAllocator&, LaunchCommand&, Event*) = 0;

  virtual void addModule(ProgramDesc) = 0;
  virtual void linkModules() = 0;
};
} // namespace athena::backend::llvm

namespace std {
template <> class hash<athena::backend::llvm::Device> {
public:
  size_t operator()(const athena::backend::llvm::Device& dev) const {
    auto hash = std::hash<std::string>()(dev.getDeviceName());
    std::cout << dev.getDeviceName() << " = " << hash << std::endl;
    return hash;
  }
};
} // namespace std
#endif // ATHENA_DEVICE_H
