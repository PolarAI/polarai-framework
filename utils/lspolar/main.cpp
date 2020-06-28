#include "RuntimeDriver.h"

#include <iostream>

using namespace athena::backend::llvm;

static std::string providerToString(DeviceProvider provider) {
  switch (provider) {
  case DeviceProvider::CUDA:
    return "CUDA";
  case DeviceProvider::SYCL:
    return "SYCL";
  case DeviceProvider::OpenCL:
    return "OpenCL";
  case DeviceProvider::VULKAN:
    return "Vulkan";
  case DeviceProvider::HOST:
    return "Host";
  }
}

static std::string kindToString(DeviceKind kind) {
  switch (kind) {
  case DeviceKind::CPU:
    return "CPU";
  case DeviceKind::GPU:
    return "GPU";
  case DeviceKind::FPGA:
    return "FPGA";
  case DeviceKind::OTHER_ACCELERATOR:
    return "Other";
  case DeviceKind::HOST:
    return "Host";
  }
}

int main() {
  athena::backend::llvm::RuntimeDriver driver;
  auto devices = driver.getDeviceList();

  std::cout << "Total device count: " << devices.size() << "\n\n";

  for (const auto& device : devices) {
    std::cout << "Name       : " << device->getDeviceName() << "\n";
    std::cout << "Provider   : " << providerToString(device->getProvider())
              << "\n";
    std::cout << "Kind       : " << kindToString(device->getKind()) << "\n";
    std::cout << "Allocator  : " << device->hasAllocator() << "\n";
    std::cout << "Partition\n";
    std::cout << "  Equally  : "
              << device->isPartitionSupported(Device::PartitionDomain::EQUALLY)
              << "\n";
    std::cout << "  By count : "
              << device->isPartitionSupported(Device::PartitionDomain::BY_COUNT)
              << "\n";
    std::cout << "  NUMA     : "
              << device->isPartitionSupported(Device::PartitionDomain::NUMA)
              << "\n";

    std::cout << "\n";
  }

  return 0;
}
