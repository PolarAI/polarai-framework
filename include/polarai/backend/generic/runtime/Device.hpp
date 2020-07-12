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
#include <polarai/backend/generic/runtime/ProgramDesc.hpp>

#include <cstddef>
#include <memory>

struct LaunchCommand;

namespace polarai::backend::generic {

class Device;
class Event;
class BackendAllocator;

enum class DeviceProvider { CUDA, HIP, SYCL, OpenCL, VULKAN, METAL, HOST };

enum class DeviceKind { CPU, GPU, FPGA, OTHER_ACCELERATOR, HOST };

class Device {
public:
  Device() = default;
  virtual ~Device() = default;

  enum class PartitionDomain { EQUALLY, BY_COUNT, NUMA };
  ///@{ \name Device information
  virtual DeviceProvider getProvider() const = 0;
  virtual DeviceKind getKind() const = 0;
  virtual std::string getDeviceName() const = 0;
  virtual bool isPartitionSupported(PartitionDomain domain) { return false; };
  virtual bool hasAllocator() { return false; };
  ///@}
  virtual std::vector<std::shared_ptr<Device>>
  partition(PartitionDomain domain) = 0;
  virtual std::shared_ptr<AllocatorLayerBase> getAllocator() = 0;

  virtual bool operator==(const Device& device) const { return false; };
  bool operator!=(const Device& device) const { return !(*this == device); };

  virtual void copyToHost(MemoryRecord record, void* dest) const = 0;
  virtual void copyToDevice(MemoryRecord record, void* src) const = 0;

  virtual Event* launch(BackendAllocator&, LaunchCommand&, Event*) = 0;

  virtual void consumeEvent(Event*) = 0;

  virtual void
  selectBinary(std::vector<std::shared_ptr<ProgramDesc>>& programs) = 0;
};
} // namespace polarai::backend::generic

namespace std {
template <> class hash<polarai::backend::generic::Device> {
public:
  size_t operator()(const polarai::backend::generic::Device& dev) const {
    auto hash = std::hash<std::string>()(dev.getDeviceName());
    return hash;
  }
};
} // namespace std
