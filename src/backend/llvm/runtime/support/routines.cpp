#include <athena/backend/llvm/BackendAllocator.h>
#include <athena/backend/llvm/runtime/Device.h>
#include <athena/backend/llvm/runtime/Event.h>
#include <athena/backend/llvm/runtime/GraphHandle.h>
#include <athena/backend/llvm/runtime/LaunchCommand.h>
#include <athena/backend/llvm/runtime/TensorInfo.h>
#include <athena/backend/llvm/runtime/support/export.h>

using namespace athena::backend::llvm;

static MemoryRecord tensorInfoToRecord(TensorInfo* tensor) {
  MemoryRecord record;
  record.virtualAddress = tensor->virtAddr;
  record.allocationSize = athena::core::sizeOfDataType(
      static_cast<athena::core::DataType>(tensor->dataType));
  for (int i = 0; i < tensor->dims; i++) {
    record.allocationSize *= tensor->shape[i];
  }
  return record;
}

extern "C" {

ATH_RT_SUPPORT_EXPORT void ath_allocate(GraphHandle* handle, Device& device,
                                        TensorInfo* tensor) {
  auto record = tensorInfoToRecord(tensor);
  handle->allocator->allocate(record, device);
}

ATH_RT_SUPPORT_EXPORT void ath_release(GraphHandle* handle, Device& device,
                                       TensorInfo* tensor) {
  auto record = tensorInfoToRecord(tensor);
  handle->allocator->release(record, device);
}

ATH_RT_SUPPORT_EXPORT void
ath_lock(GraphHandle* handle, Device& device, TensorInfo* tensor,
                athena::core::internal::LockType type) {
  auto record = tensorInfoToRecord(tensor);
  handle->allocator->lock(record, device, type);
}

ATH_RT_SUPPORT_EXPORT Device* ath_device_select(GraphHandle* handle,
                                                uint64_t nodeId) {
  return nullptr;
}

ATH_RT_SUPPORT_EXPORT Event* ath_launch(Device& device,
                                        BackendAllocator& allocator,
                                        Event* event, LaunchCommand& command) {
  return device.launch(allocator, command, event);
}
}
