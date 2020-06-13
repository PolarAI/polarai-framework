#pragma once

#include <athena/backend/llvm/runtime/Device.h>

namespace athena::backend::llvm {
class RuntimeDriver {
public:
  RuntimeDriver();
  ~RuntimeDriver();
  auto getDeviceList() -> std::vector<std::shared_ptr<Device>>& {
    return mDevices;
  };

private:
  std::vector<std::shared_ptr<Device>> mDevices;
  std::vector<std::pair<DeviceContainer, std::function<void(DeviceContainer)>>>
      mRecyclers;
};
} // namespace athena::backend::llvm
