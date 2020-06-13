#include "RuntimeDriver.h"
#include "config.h"

#include <athena/backend/llvm/runtime/Device.h>

#include <llvm/Support/DynamicLibrary.h>

namespace athena::backend::llvm {
RuntimeDriver::RuntimeDriver() {
  auto libraries = getListOfLibraries();

  for (auto lib : libraries) {
    ::llvm::sys::DynamicLibrary dynLib =
        ::llvm::sys::DynamicLibrary::getPermanentLibrary(lib.c_str());
    if (!dynLib.isValid())
      continue;

    void* listDevPtr = dynLib.getAddressOfSymbol("getAvailableDevices");
    auto listDevFunc = reinterpret_cast<DeviceContainer (*)()>(listDevPtr);

    auto externalDevices = listDevFunc();
    for (int i = 0; i < externalDevices.count; i++) {
      mDevices.push_back(&externalDevices.devices[i]);
    }
  }
}
} // namespace athena::backend::llvm
