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

#include <polarai/backend/generic/runtime/Device.hpp>

namespace polarai::backend::generic {
class HostDevice : public Device {
public:
  HostDevice() = default;
  ///@{ \name Device information
  auto getProvider() const -> DeviceProvider override {
    return DeviceProvider::HOST;
  }
  auto getKind() const -> DeviceKind override { return DeviceKind::HOST; }
  std::string getDeviceName() const override { return "host"; }
  bool isPartitionSupported(PartitionDomain domain) override { return false; }
  bool hasAllocator() override { return false; }
  ///@}
  std::vector<std::shared_ptr<Device>>
  partition(PartitionDomain domain) override {
    return std::vector<std::shared_ptr<Device>>{};
  }
  std::shared_ptr<AllocatorLayerBase> getAllocator() override {
    return nullptr;
  }

  bool operator==(const Device& device) const override {
    return device.getDeviceName() == getDeviceName();
  }

  void copyToHost(MemoryRecord record, void* dest) const override {}
  void copyToDevice(MemoryRecord record, void* src) const override {}

  Event* launch(BackendAllocator&, LaunchCommand&, Event*) override {
    return nullptr;
  }

  void consumeEvent(Event*) override {}

  void
  selectBinary(std::vector<std::shared_ptr<ProgramDesc>>& programs) override{};
};
} // namespace polarai::backend::llvm
