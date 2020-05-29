#include "RuntimeDriver.h"

int main() {
  athena::backend::llvm::RuntimeDriver driver;
  auto& devices = driver.getDeviceList();

  return 0;
}
