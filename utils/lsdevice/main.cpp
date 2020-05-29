#include "RuntimeDriver.h"

#include <iostream>

int main() {
  athena::backend::llvm::RuntimeDriver driver;
  auto& devices = driver.getDeviceList();

  std::cout << "Total device count: " << devices.size() << "\n";

  return 0;
}
