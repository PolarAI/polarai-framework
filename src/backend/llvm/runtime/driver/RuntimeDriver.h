#pragma once

#include <athena/backend/llvm/runtime/Device.h>

namespace athena::backend::llvm {
class RuntimeDriver {
public:
  RuntimeDriver();
  auto getDeviceList() -> std::vector<Device*>& { return mDevices; };

private:
  std::vector<Device*> mDevices;
};
} // namespace athena::backend::llvm
