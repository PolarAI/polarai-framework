//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#pragma once

#include <polarai/backend/generic/BackendAllocator.hpp>
#include <polarai/backend/generic/runtime/BackendAccessor.hpp>
#include <polarai/backend/generic/runtime/Device.hpp>
#include <polarai/backend/generic/runtime/Event.hpp>
#include <polarai/backend/generic/runtime/GraphHandle.hpp>
#include <polarai/backend/generic/runtime/LaunchCommand.h>
#include <polarai/backend/generic/runtime/TensorInfo.h>
#include <polar_rt_support_export.h>

#include <cstddef>

extern "C" {
POLAR_RT_SUPPORT_EXPORT void ath_allocate(GraphHandle* handle, polarai::backend::generic::Device& device,
                                          TensorInfo* tensor);
POLAR_RT_SUPPORT_EXPORT void ath_release(GraphHandle* handle, polarai::backend::generic::Device& device,
                                         TensorInfo* tensor,
                                         polarai::backend::generic::Event* blockingEvt);
POLAR_RT_SUPPORT_EXPORT void ath_lock(GraphHandle* handle, polarai::backend::generic::Device& device,
                                      TensorInfo* tensor,
                                      polarai::core::internal::LockType type);
POLAR_RT_SUPPORT_EXPORT polarai::backend::generic::Device* ath_device_select(GraphHandle* handle,
                                                  uint64_t nodeId);
POLAR_RT_SUPPORT_EXPORT void ath_barrier(uint32_t count, polarai::backend::generic::Event** events);
POLAR_RT_SUPPORT_EXPORT polarai::backend::generic::Event* ath_launch(GraphHandle* handle, polarai::backend::generic::Device* device,
                                          polarai::backend::generic::Event* event,
                                          LaunchCommand& command);
POLAR_RT_SUPPORT_EXPORT void ath_load(GraphHandle* handle, uint64_t nodeId,
                                      TensorInfo* tensor);
}