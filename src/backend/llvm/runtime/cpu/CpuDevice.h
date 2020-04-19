/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_CPUDEVICE_H
#define ATHENA_CPUDEVICE_H

#include <athena/backend/llvm/runtime/Device.h>

#include "CpuQueue.h"
#include <athena/core/inner/Tensor.h>
#include <memory>

namespace athena::backend::llvm {
class CPUDevice : public Device {
private:
  CpuQueue mQueue;

public:
  bool isPartitionSupported(PartitionDomain domain) override;
  DeviceContainer partition(PartitionDomain domain) override;

  std::string getDeviceName() const override { return "CPU"; }

  bool operator==(const Device& device) const override {
    return getDeviceName() == device.getDeviceName();
  }

  Queue& getQueue() override { return mQueue; }
};
} // namespace athena::backend::llvm

#endif // ATHENA_CPUDEVICE_H
