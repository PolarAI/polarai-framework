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

#include <athena/backend/llvm/runtime/api.h>
#include <gtest/gtest.h>
#include <llvm/Support/DynamicLibrary.h>
#include "../../../src/backend/llvm/allocators/LayerAllocator.h"
#include <athena/backend/llvm/runtime/LaunchCommand.h>
#include <athena/core/context/Context.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using initContextPtr = void (*)(athena::backend::llvm::DeviceContainer);
using getAvailableDevicesPtr = athena::backend::llvm::DeviceContainer (*)();

class RuntimeTestBase : public testing::Test {
protected:
  getAvailableDevicesPtr getAvailableDevicesFunc{};

  void SetUp() override {
    std::string errStr;
    auto dynLib = llvm::sys::DynamicLibrary::getPermanentLibrary(
        getenv("RUNTIME_LIB"), &errStr);
    if (!errStr.empty()) {
      std::cerr << errStr;
      std::terminate();
    }

    void* getAvailableDevicesRaw =
        dynLib.getAddressOfSymbol("getAvailableDevices");
    getAvailableDevicesFunc =
        reinterpret_cast<getAvailableDevicesPtr>(getAvailableDevicesRaw);
  }

public:
  virtual size_t getAvailableDeviceCount() = 0;
  virtual std::vector<char> getSampleProgram() = 0;
};

class OpenCLRuntimeTest : public RuntimeTestBase {
public:
  size_t getAvailableDeviceCount() override {
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    std::vector<cl_platform_id> platforms;
    platforms.resize(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    size_t deviceCount = 0;
    for (auto platform : platforms) {
      cl_uint platformDevCount;
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr,
                     &platformDevCount);
      deviceCount += platformDevCount;
    }

    return deviceCount;
  }
  std::vector<char> getSampleProgram() override {
    std::string program = R"(
kernel void add(global float* a, float b) {
  int globalId = get_global_id(0);
  a[globalId] += b;
}
)";
    std::vector<char> data(program.begin(), program.end());
    return data;
  }
};

TEST_F(OpenCLRuntimeTest, FindsAllDevices) {
  auto rtDevices = getAvailableDevicesFunc();

  const auto devCount = getAvailableDeviceCount();

  ASSERT_EQ(devCount, rtDevices.count);
}

TEST_F(OpenCLRuntimeTest, ExecutesSimpleKernel) {
  using namespace athena::core;
  using namespace athena::backend::llvm;

  auto deviceContainer = getAvailableDevicesFunc();

  for (size_t i = 0; i < deviceContainer.count; i++) {
    LayerAllocator allocator;
    Device& device = deviceContainer.devices[i];

    allocator.registerDevice(device);

    Context ctx;
    auto ctxInternalPtr = ctx.internal();
    auto tensorIndex = ctxInternalPtr->create<TensorInternal>(
        ctxInternalPtr, ctxInternalPtr->getNextPublicIndex(), DataType::FLOAT,
        TensorShape{30});
    auto& tensor = ctxInternalPtr->getRef<TensorInternal>(tensorIndex);

    allocator.allocate(tensor);
    allocator.lock(tensor, LockType::READ_WRITE);
    auto data = static_cast<float*>(allocator.get(tensor));
    data[0] = 1;
    data[1] = 2;
    data[2] = 3;
    allocator.release(tensor);

    auto program = getSampleProgram();
    ProgramDesc programDesc;
    programDesc.data = program.data();
    programDesc.length = program.size();
    programDesc.type = ProgramDesc::TEXT;

    device.addModule(programDesc);
    device.linkModules();

    float b = 10;

    ArgDesc args[2];
    args[0].type = ArgDesc::TENSOR;
    args[0].size = 0;
    args[0].arg = &tensor;
    args[1].type = ArgDesc::DATA;
    args[1].size = sizeof(float);
    args[1].arg = &b;

    size_t count = 3;

    LaunchCommand command;
    command.kernelName = "add";
    command.argsCount = 2;
    command.args = reinterpret_cast<ArgDesc*>(&args);
    command.workDim = 1;
    command.globalSize = &count;
    command.localSize = &count;

    allocator.lock(tensor, deviceContainer.devices[i], LockType::READ_WRITE);
    device.launch(allocator, command, nullptr);
    device.getQueue().wait();
    allocator.release(tensor, deviceContainer.devices[i]);

    allocator.lock(tensor, LockType::READ);
    auto result = static_cast<float*>(allocator.get(tensor));
    allocator.release(tensor);
    EXPECT_FLOAT_EQ(result[0], 11);
    EXPECT_FLOAT_EQ(result[1], 12);
    EXPECT_FLOAT_EQ(result[2], 13);
  }
}
