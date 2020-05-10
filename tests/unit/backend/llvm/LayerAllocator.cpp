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
#include <athena/core/inner/Tensor.h>

#include <athena/core/Context.h>
#include <gtest/gtest.h>

using namespace athena::backend::llvm;
using namespace athena::core;

class MockQueue : public Queue {
public:
  void wait() override {}
};

class MockDevice : public Device {
private:
  MockQueue mQueue;

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
  Queue& getQueue() override { return mQueue; }
};

TEST(LLVMBackend, LayerAllocatorSimple) {
  LayerAllocator allocator;

  Context ctx;
  inner::Tensor tensor(DataType::FLOAT, {30}, ctx);

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
  inner::Tensor tensor(DataType::FLOAT, {30}, ctx);

  allocator.allocate(tensor, device);
  auto ptr = allocator.get<void*>(tensor, device);
  ASSERT_NE(ptr, nullptr);
}

TEST(LLVMBackend, LayerAllocatorDeviceDoubleLock) {
  LayerAllocator allocator;
  MockDevice device;

  allocator.registerDevice(device);

  Context ctx;
  inner::Tensor tensor(DataType::FLOAT, {30}, ctx);

  allocator.allocate(tensor, device);
  allocator.lock(tensor, device, LockType::READ);
  ASSERT_DEATH(allocator.lock(tensor, device, LockType::READ_WRITE),
               "Attempt get READ_WRITE lock for tensor that is already locked");
}
