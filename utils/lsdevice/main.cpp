#include "RuntimeDriver.h"

#include <iostream>

int main() {
  athena::backend::llvm::RuntimeDriver driver;
  auto& devices = driver.getDeviceList();

  std::cout << "Total device count: " << devices.size() << "\n\n";

  for (auto* device : devices) {
    std::cout << "------------------"
              << "\n";
    std::cout << "Device name: " << device->getDeviceName() << "\n";

    std::cout << "\n";
  }

  return 0;
}
