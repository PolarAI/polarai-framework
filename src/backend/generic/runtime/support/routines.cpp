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

#include "../utils/utils.hpp"
#include "api.hpp"

#include <polarai/backend/generic/BackendAllocator.hpp>
#include <polarai/backend/generic/runtime/BackendAccessor.hpp>
#include <polarai/backend/generic/runtime/Device.hpp>
#include <polarai/backend/generic/runtime/Event.hpp>
#include <polarai/backend/generic/runtime/GraphHandle.hpp>
#include <polarai/backend/generic/runtime/LaunchCommand.h>
#include <polarai/backend/generic/runtime/TensorInfo.h>

#ifdef BUILD_DLL
#undef BUILD_DLL
#endif
#include <polarai/core/loader/internal/AbstractLoaderInternal.hpp>
#include <polarai/utils/error/FatalError.hpp>
#include <polar_rt_support_export.h>

#include <iostream>

using namespace polarai::backend::generic;

extern "C" {

void ath_allocate(GraphHandle* handle, Device& device,
                                          TensorInfo* tensor) {
  auto record = tensorInfoToRecord(tensor);
  if (device.getDeviceName() == "host") {
    handle->allocator->allocate(record);
  } else {
    handle->allocator->allocate(record, device);
  }
}

void ath_release(GraphHandle* handle, Device& device,
                                         TensorInfo* tensor,
                                         Event* blockingEvt) {
  auto record = tensorInfoToRecord(tensor);
  if (device.getDeviceName() == "host") {
    handle->allocator->release(record);
  } else if (blockingEvt) {
    blockingEvt->addCallback([handle, &device, record]() {
      handle->allocator->release(record, device);
    });
  } else {
    handle->allocator->release(record, device);
  }
}

void ath_lock(GraphHandle* handle, Device& device,
                                      TensorInfo* tensor,
                                      polarai::core::internal::LockType type) {
  auto record = tensorInfoToRecord(tensor);
  if (device.getDeviceName() == "host") {
    handle->allocator->lock(record, type);
  } else {
    handle->allocator->lock(record, device, type);
  }
}

Device* ath_device_select(GraphHandle* handle,
                                                  uint64_t nodeId) {
  if (handle->isHostNode.count(nodeId)) {
    return handle->devices.back().get();
  }
  return handle->devices.front().get(); // TODO real device selection logic.
}

void ath_barrier(uint32_t count, Event** events) {
  for (int i = 0; i < count; i++) {
    if (events[i]) {
      events[i]->wait();
    }
  }
}

Event* ath_launch(GraphHandle* handle, Device* device,
                                          Event* event,
                                          LaunchCommand& command) {
  return device->launch(*handle->allocator, command, event);
}

void ath_load(GraphHandle* handle, uint64_t nodeId,
                                      TensorInfo* tensor) {
  auto* loader = handle->mLoaders[nodeId];
  auto record = tensorInfoToRecord(tensor);
  auto* ptr = handle->allocator->get(record);
  auto dataType = static_cast<polarai::core::DataType>(tensor->dataType);
  if (dataType == polarai::core::DataType::FLOAT) {
    BackendAccessor<float> acc(static_cast<float*>(ptr), tensor->dims,
                               tensor->shape);
    loader->load(acc);
  } else if (dataType == polarai::core::DataType::DOUBLE) {
    polarai::utils::FatalError(polarai::utils::ATH_FATAL_OTHER,
                              "Double is not supported.");
    //    BackendAccessor<double> acc(static_cast<double*>(ptr), tensor->dims,
    //                                tensor->shape);
    //    loader->load(acc);
  }
}
}
