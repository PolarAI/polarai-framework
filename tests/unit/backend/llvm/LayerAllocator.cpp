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

#include "../../../../src/backend/llvm/allocators/LayerAllocator.h"
#include <athena/backend/llvm/runtime/Device.h>
#include <athena/core/tensor/internal/TensorInternal.h>

#include <athena/core/context/Context.h>
#include <gtest/gtest.h>

using namespace athena::backend::llvm;
using namespace athena::core;

class MockDevice : public Device {
public:
  std::string getDeviceName() const override { return "Mock"; }
  bool isPartitionSupported(PartitionDomain domain) override { return false; }
  DeviceContainer partition(PartitionDomain domain) override {
    return DeviceContainer{};
  }
  bool hasAllocator() override { return false; }
  std::shared_ptr<AllocatorLayerBase> getAllocator() override {
    return nullptr;
  }
  bool operator==(const Device& device) const override {
    return device.getDeviceName() == getDeviceName();
  }
  void copyToHost(const internal::TensorInternal& tensor,
                  void* dest) const override{};
  void copyToHost(MemoryRecord record, void* dest) const override{};
  void copyToDevice(const internal::TensorInternal& tensor,
                    void* src) const override{};
  void copyToDevice(MemoryRecord record, void* src) const override{};
  Event* launch(BackendAllocator&, LaunchCommand&, Event*) override {
    return nullptr;
  };
  void addModule(ProgramDesc) override{}
  void linkModules() override{}
  void consumeEvent(Event*) override {}
};

TEST(LLVMBackend, LayerAllocatorSimple) {
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

TEST(LLVMBackend, LayerAllocatorDevice) {
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
