/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_DEVICE_H
#define ATHENA_DEVICE_H

#include <athena/backend/llvm/AllocatorLayerBase.h>
#include <athena/backend/llvm/MemoryRecord.h>
#include <athena/backend/llvm/llvm_export.h>
#include <athena/core/inner/Tensor.h>

#include <cstddef>
#include <memory>

namespace athena::backend::llvm {

class Device;

extern "C" struct ATH_BACKEND_LLVM_EXPORT DeviceContainer {
  Device* devices;
  size_t count;
};

class ATH_BACKEND_LLVM_EXPORT Device {
public:
  Device() = default;

  enum class PartitionDomain { EQUALLY, BY_COUNT, NUMA };
  virtual std::string getDeviceName() const = 0;
  virtual bool isPartitionSupported(PartitionDomain domain) { return false; };
  virtual DeviceContainer partition(PartitionDomain domain) {
    return DeviceContainer{};
  };
  virtual bool hasAllocator() { return false; };
  virtual std::shared_ptr<AllocatorLayerBase> getAllocator() {
    return nullptr;
  };

  virtual bool operator==(const Device& device) const { return false; };
  bool operator!=(const Device& device) const { return !(*this == device); };

  virtual void copyToHost(const core::inner::Tensor& tensor,
                          void* dest) const {};
  virtual void copyToHost(MemoryRecord record, void* dest) const {};
  virtual void copyToDevice(const core::inner::Tensor& tensor,
                            void* src) const {};
  virtual void copyToDevice(MemoryRecord record, void* src) const {};
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
