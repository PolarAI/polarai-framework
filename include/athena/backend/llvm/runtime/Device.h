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

#ifndef ATHENA_DEVICE_H
#define ATHENA_DEVICE_H

#include <athena/backend/llvm/llvm_export.h>

#include <cstddef>

namespace athena::backend::llvm {

class Device;

extern "C" struct ATH_BACKEND_LLVM_EXPORT DeviceContainer {
  Device* devices;
  size_t count;
};

class ATH_BACKEND_LLVM_EXPORT Device {
public:
  enum class PartitionDomain { EQUALLY, BY_COUNT, NUMA };
  virtual bool isPartitionSupported(PartitionDomain domain) = 0;
  virtual DeviceContainer partition(PartitionDomain domain) = 0;
};
} // namespace athena::backend::llvm

#endif // ATHENA_DEVICE_H
