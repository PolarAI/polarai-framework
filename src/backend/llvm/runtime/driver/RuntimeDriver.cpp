#include "RuntimeDriver.h"
#include "config.h"

#include <athena/backend/llvm/runtime/Device.h>

#include <llvm/Support/DynamicLibrary.h>

#include <iostream>
#include <string>

namespace athena::backend::llvm {
RuntimeDriver::RuntimeDriver() {
  auto libraries = getListOfLibraries();

  for (auto lib : libraries) {
    std::string errMsg;
    ::llvm::sys::DynamicLibrary dynLib =
        ::llvm::sys::DynamicLibrary::getPermanentLibrary(lib.c_str(), &errMsg);
    if (!dynLib.isValid()) {
      std::cerr << errMsg << "\n"; // fixme use in-house stream
      continue;
    }

    void* initCtxPtr = dynLib.getAddressOfSymbol("initContext");
    auto initCtxFunc = reinterpret_cast<Context* (*)()>(initCtxPtr);
    void* closeCtxPtr = dynLib.getAddressOfSymbol("closeContext");
    auto closeCtxFunc = reinterpret_cast<void (*)(Context*)>(closeCtxPtr);

    Context* ctx = initCtxFunc();
    mContexts.emplace_back(ctx,
                           [closeCtxFunc](Context* ctx) { closeCtxFunc(ctx); });
    auto &newDevs = ctx->getDevices();
    mDevices.insert(mDevices.end(), newDevs.begin(), newDevs.end());
  }
}

} // namespace athena::backend::llvm
