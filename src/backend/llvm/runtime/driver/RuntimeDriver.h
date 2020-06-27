#pragma once

#include <athena/backend/llvm/runtime/Device.h>
#include <athena/backend/llvm/runtime/Context.h>

namespace athena::backend::llvm {
class RuntimeDriver {
public:
  RuntimeDriver();
  auto getDeviceList() -> std::vector<std::shared_ptr<Device>>& {
    return mDevices;
  };

private:
  std::vector<std::shared_ptr<Device>> mDevices;
  std::vector<std::shared_ptr<Context>> mContexts;
};
} // namespace athena::backend::llvm
