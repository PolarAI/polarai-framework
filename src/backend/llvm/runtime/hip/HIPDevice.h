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

#include <athena/backend/llvm/runtime/Device.h>

#include <hip/hip_runtime.h>
#include <string>

namespace athena::backend::llvm {
class HIPDevice : public Device {
public:
  HIPDevice(hipDevice_t device);
  auto getProvider() const -> DeviceProvider override {
    return DeviceProvider::HIP;
  }
  auto getKind() const -> DeviceKind override { return DeviceKind::GPU; }
  std::string getDeviceName() const override;
  bool isPartitionSupported(PartitionDomain domain) override { return false; }
  bool hasAllocator() override { return false; }

  std::vector<std::shared_ptr<Device>>
  partition(PartitionDomain domain) override {
    return std::vector<std::shared_ptr<Device>>{};
  };
  std::shared_ptr<AllocatorLayerBase> getAllocator() override {
    return nullptr;
  };

  bool operator==(const Device& device) const override {
    return mDeviceName == device.getDeviceName();
  };

  void copyToHost(const core::internal::TensorInternal& tensor,
                  void* dest) const override{};
  void copyToHost(MemoryRecord record, void* dest) const override{};
  void copyToDevice(const core::internal::TensorInternal& tensor,
                    void* src) const override{};
  void copyToDevice(MemoryRecord record, void* src) const override{};

  Event* launch(BackendAllocator&, LaunchCommand&, Event*) override {
    return nullptr;
  };

  void addModule(ProgramDesc) override{};
  void linkModules() override{};

  void consumeEvent(Event*) override{};

private:
  hipDevice_t mDevice;
  std::string mDeviceName;
};
} // namespace athena::backend::llvm
