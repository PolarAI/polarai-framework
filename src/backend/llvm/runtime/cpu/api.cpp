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

#include "CpuDevice.h"

#include <athena/backend/llvm/runtime/api.h>

using namespace athena::backend::llvm;

static CPUDevice kCpuDevice;

DeviceContainer getAvailableDevices() {
  DeviceContainer deviceContainer{&kCpuDevice, 1};
  return deviceContainer;
}
void initializeContext(athena::backend::llvm::DeviceContainer) {}
void releaseContext() {}