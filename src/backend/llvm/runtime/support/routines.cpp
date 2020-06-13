#include "../utils/utils.h"

#include <athena/backend/llvm/BackendAllocator.h>
#include <athena/backend/llvm/runtime/BackendAccessor.h>
#include <athena/backend/llvm/runtime/Device.h>
#include <athena/backend/llvm/runtime/Event.h>
#include <athena/backend/llvm/runtime/GraphHandle.h>
#include <athena/backend/llvm/runtime/LaunchCommand.h>
#include <athena/backend/llvm/runtime/TensorInfo.h>
#include <athena/backend/llvm/runtime/support/export.h>
#include <athena/core/loader/internal/AbstractLoaderInternal.h>

#include <iostream>

using namespace athena::backend::llvm;

extern "C" {

ATH_RT_SUPPORT_EXPORT void ath_allocate(GraphHandle* handle, Device& device,
                                        TensorInfo* tensor) {
  auto record = tensorInfoToRecord(tensor);
  if (device.getDeviceName() == "host") {
    handle->allocator->allocate(record);
  } else {
    handle->allocator->allocate(record, device);
  }
}

ATH_RT_SUPPORT_EXPORT void ath_release(GraphHandle* handle, Device& device,
                                       TensorInfo* tensor) {
  auto record = tensorInfoToRecord(tensor);
  if (device.getDeviceName() == "host") {
    handle->allocator->allocate(record);
  } else {
    handle->allocator->release(record, device);
  }
}

ATH_RT_SUPPORT_EXPORT void ath_lock(GraphHandle* handle, Device& device,
                                    TensorInfo* tensor,
                                    athena::core::internal::LockType type) {
  auto record = tensorInfoToRecord(tensor);
  if (device.getDeviceName() == "host") {
    handle->allocator->lock(record, type);
  } else {
    handle->allocator->lock(record, device, type);
  }
}

ATH_RT_SUPPORT_EXPORT Device* ath_device_select(GraphHandle* handle,
                                                uint64_t nodeId) {
  if (handle->isHostNode.count(nodeId)) {
    return handle->devices.back();
  }
  return handle->devices.front(); // TODO real device selection logic.
}

ATH_RT_SUPPORT_EXPORT void ath_barrier(uint32_t count, Event** events) {}

ATH_RT_SUPPORT_EXPORT Event* ath_launch(GraphHandle* handle, Device* device,
                                        Event* event, LaunchCommand& command) {
  return device->launch(*handle->allocator, command, event);
}

ATH_RT_SUPPORT_EXPORT void ath_load(GraphHandle* handle, uint64_t nodeId,
                                    TensorInfo* tensor) {
  auto* loader = handle->mLoaders[nodeId];
  auto record = tensorInfoToRecord(tensor);
  auto* ptr = handle->allocator->get(record);
  auto dataType = static_cast<athena::core::DataType>(tensor->dataType);
  if (dataType == athena::core::DataType::FLOAT) {
    BackendAccessor<float> acc(static_cast<float*>(ptr), tensor->dims,
                               tensor->shape);
    loader->load(acc);
  } else if (dataType == athena::core::DataType::DOUBLE) {
    BackendAccessor<double> acc(static_cast<double*>(ptr), tensor->dims,
                                tensor->shape);
    loader->load(acc);
  }
}
}
