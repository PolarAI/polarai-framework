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

#ifndef ATHENA_HOSTDEVICE_H
#define ATHENA_HOSTDEVICE_H

#include <athena/backend/llvm/runtime/Device.h>

namespace athena::backend::llvm {
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

  void copyToHost(const core::internal::TensorInternal& tensor,
                  void* dest) const override {}
  void copyToHost(MemoryRecord record, void* dest) const override {}
  void copyToDevice(const core::internal::TensorInternal& tnsor,
                    void* src) const override {}
  void copyToDevice(MemoryRecord record, void* src) const override {}

  Event* launch(BackendAllocator&, LaunchCommand&, Event*) override {
    return nullptr;
  }

  void consumeEvent(Event*) override {}

  void
  selectBinary(std::vector<std::shared_ptr<ProgramDesc>>& programs) override{};
};
} // namespace athena::backend::llvm

#endif // ATHENA_HOSTDEVICE_H
