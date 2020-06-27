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

#include <cuda.h>
#include <string>

namespace athena::backend::llvm {
class CudaDevice : public Device {
public:
  CudaDevice(CUdevice device);
  std::string getDeviceName() const override;
  bool isPartitionSupported(PartitionDomain domain) override { return false; }
  bool hasAllocator() override { return false; }

  DeviceContainer partition(PartitionDomain domain) override {
    return DeviceContainer{};
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
  CUdevice mDevice;
  std::string mDeviceName;
};
} // namespace athena::backend::llvm
