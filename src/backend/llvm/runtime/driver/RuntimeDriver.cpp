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

    void* consumeDevPtr = dynLib.getAddressOfSymbol("consumeDevice");
    auto consumeDevFunc = reinterpret_cast<void (*)(Device*)>(consumeDevPtr);

    auto externalDevices = listDevFunc();
    for (int i = 0; i < externalDevices.count; i++) {
      mDevices.emplace_back(
          externalDevices.devices[i],
          [consumeDevFunc](Device* dev) { consumeDevFunc(dev); });
    }

    void* consumeContPtr = dynLib.getAddressOfSymbol("consumeContainer");
    auto consumeContFunc = reinterpret_cast<void (*)(Device*)>(consumeContPtr);
  }
}

RuntimeDriver::~RuntimeDriver() {
  for (const auto& consumer : mRecyclers) {
    consumer.second(consumer.first);
  }
}
} // namespace athena::backend::llvm
