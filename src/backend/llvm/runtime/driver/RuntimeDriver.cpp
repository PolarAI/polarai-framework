#include "RuntimeDriver.h"
#include "config.h"

#include <llvm/Support/DynamicLibrary.h>

namespace athena::backend::llvm {
RuntimeDriver::RuntimeDriver() {
  auto libraries = getListOfLibraries();

  for (auto lib : libraries) {
    ::llvm::sys::DynamicLibrary dynLib =
        ::llvm::sys::DynamicLibrary::getPermanentLibrary(lib.c_str());
    void* listDevPtr = dynLib.getAddressOfSymbol("listDevices");
    auto listDevFunc = reinterpret_cast<std::vector<Device*> (*)()>(listDevPtr);

    auto externalDevices = listDevFunc();
    mDevices.insert(mDevices.end(), externalDevices.begin(),
                    externalDevices.end());
  }
}
} // namespace athena::backend::llvm
