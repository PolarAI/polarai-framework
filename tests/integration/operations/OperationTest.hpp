#pragma once

#include <polarai/backend/generic/GenericExecutor.hpp>
#include <runtime/driver/RuntimeDriver.hpp>

#include <gtest/gtest.h>

class OperationTest : public ::testing::Test {
private:
  using CallbackT =
      std::function<void(polarai::backend::generic::GenericExecutor&)>;

protected:
  polarai::backend::generic::RuntimeDriver mDriver{false};

  void SetUp() override {
    if (mDriver.getDeviceList().empty()) {
      GTEST_SKIP() << "No available devices";
    }
  }

  void withEachDeviceDo(const CallbackT& cb) {
    for (auto& device : mDriver.getDeviceList()) {
      auto selector =
          [&](std::shared_ptr<polarai::backend::generic::Device>& dev) {
            return dev->getProvider() == device->getProvider() &&
                   dev->getDeviceName() == device->getDeviceName();
          };

      polarai::backend::generic::GenericExecutor executor(true, selector);
      cb(executor);
    }
  }
};
