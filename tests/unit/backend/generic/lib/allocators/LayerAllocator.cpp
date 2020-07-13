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

#include "lib/allocators/LayerAllocator.hpp"
#include <polarai/backend/generic/runtime/Device.hpp>
#include <polarai/core/tensor/internal/TensorInternal.hpp>
#include <polarai/core/context/Context.hpp>

#include <gtest/gtest.h>

using namespace polarai::backend::generic;
using namespace polarai::core;

class MockDevice : public Device {
public:
  [[nodiscard]] auto getProvider() const -> DeviceProvider override {
    return DeviceProvider::HOST;
  }
  auto getKind() const -> DeviceKind override { return DeviceKind::HOST; }
  std::string getDeviceName() const override { return "Mock"; }
  bool isPartitionSupported(PartitionDomain domain) override { return false; }
  std::vector<std::shared_ptr<Device>>
  partition(PartitionDomain domain) override {
    return std::vector<std::shared_ptr<Device>>{};
  }
  bool hasAllocator() override { return false; }
  std::shared_ptr<AllocatorLayerBase> getAllocator() override {
    return nullptr;
  }
  bool operator==(const Device& device) const override {
    return device.getDeviceName() == getDeviceName();
  }
  void copyToHost(MemoryRecord record, void* dest) const override{};
  void copyToDevice(MemoryRecord record, void* src) const override{};
  Event* launch(BackendAllocator&, LaunchCommand&, Event*) override {
    return nullptr;
  };
  void consumeEvent(Event*) override {}
  void
  selectBinary(std::vector<std::shared_ptr<ProgramDesc>>& programs) override{};
};

TEST(GenericBackendAllocators, LayerAllocatorSimple) {
  LayerAllocator allocator;

  Context ctx;
  auto ctxInternalPtr = ctx.internal();
  auto tensorIndex = ctxInternalPtr->create<TensorInternal>(
      ctxInternalPtr, ctxInternalPtr->getNextPublicIndex(), DataType::FLOAT,
      TensorShape{30});
  auto& tensor = ctxInternalPtr->getRef<TensorInternal>(tensorIndex);
  allocator.allocate(tensor);
  auto ptr = allocator.get(tensor);
  ASSERT_NE(ptr, nullptr);

  allocator.deallocate(tensor);
}

TEST(GenericBackendAllocators, LayerAllocatorDevice) {
  LayerAllocator allocator;
  MockDevice device;

  allocator.registerDevice(device);

  Context ctx;
  auto ctxInternalPtr = ctx.internal();
  auto tensorIndex = ctxInternalPtr->create<TensorInternal>(
      ctxInternalPtr, ctxInternalPtr->getNextPublicIndex(), DataType::FLOAT,
      TensorShape{30});
  auto& tensor = ctxInternalPtr->getRef<TensorInternal>(tensorIndex);

  allocator.allocate(tensor, device);
  auto ptr = allocator.get<void*>(tensor, device);
  ASSERT_NE(ptr, nullptr);
}

TEST(LLVMBackend, LayerAllocatorDeviceDoubleLock) {
  LayerAllocator allocator;
  MockDevice device;

  allocator.registerDevice(device);

  Context ctx;
  auto ctxInternalPtr = ctx.internal();
  auto tensorIndex = ctxInternalPtr->create<TensorInternal>(
      ctxInternalPtr, ctxInternalPtr->getNextPublicIndex(), DataType::FLOAT,
      TensorShape{30});
  auto& tensor = ctxInternalPtr->getRef<TensorInternal>(tensorIndex);

  allocator.allocate(tensor, device);
  allocator.lock(tensor, device, LockType::READ);
  ASSERT_DEATH(allocator.lock(tensor, device, LockType::READ_WRITE),
               "Attempt get READ_WRITE lock for tensor that is already locked");
}
