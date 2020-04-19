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

#ifndef ATHENA_API_H
#define ATHENA_API_H

#include <athena/backend/llvm/BackendAllocator.h>
#include <athena/backend/llvm/runtime/Device.h>
#include <athena/backend/llvm/runtime/LaunchCommand.h>
#include <athena/backend/llvm/runtime/ProgramDesc.h>

extern "C" {
athena::backend::llvm::DeviceContainer getAvailableDevices();
void initializeContext(athena::backend::llvm::DeviceContainer);
void releaseContext();

void addProgram(athena::backend::llvm::Device*, ProgramDesc);
void linkPrograms(athena::backend::llvm::Device*);
void launch(athena::backend::llvm::Device*,
            athena::backend::llvm::BackendAllocator*, LaunchCommand);
};

#endif // ATHENA_API_H
