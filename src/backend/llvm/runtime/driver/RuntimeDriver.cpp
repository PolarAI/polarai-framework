#include "RuntimeDriver.h"
#include "config.h"

#include <athena/backend/llvm/runtime/Device.h>

#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/raw_ostream.h>

namespace athena::backend::llvm {
RuntimeDriver::RuntimeDriver() {
  auto libraries = getListOfLibraries();

  for (const auto& lib : libraries) {
    std::string errMsg;
    ::llvm::sys::DynamicLibrary dynLib =
        ::llvm::sys::DynamicLibrary::getPermanentLibrary(lib.c_str(), &errMsg);
    if (!dynLib.isValid()) {
      // fixme use Athena logger
      ::llvm::errs() << "Warning: failed to load " << lib << "\n";
      ::llvm::errs() << errMsg << "\n";
      continue;
    }

    void* listDevPtr = dynLib.getAddressOfSymbol("getAvailableDevices");
    auto listDevFunc = reinterpret_cast<DeviceContainer (*)()>(listDevPtr);

    auto externalDevices = listDevFunc();
    for (int i = 0; i < externalDevices.count; i++) {
      mDevices.push_back(&externalDevices.devices[i]);
    }
  }
}
} // namespace athena::backend::llvm
