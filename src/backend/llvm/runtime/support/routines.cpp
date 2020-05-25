#include <athena/backend/llvm/BackendAllocator.h>
#include <athena/backend/llvm/runtime/Device.h>
#include <athena/backend/llvm/runtime/Event.h>
#include <athena/backend/llvm/runtime/LaunchCommand.h>
#include <athena/backend/llvm/runtime/support/export.h>

using namespace athena::backend::llvm;

extern "C" {

ATH_RT_SUPPORT_EXPORT void ath_allocate(Device& device,
                                        BackendAllocator& allocator,
                                        MemoryRecord& record) {
  allocator.allocate(record, device);
}

ATH_RT_SUPPORT_EXPORT void ath_release_tensor(Device& device,
                                              BackendAllocator& allocator,
                                              MemoryRecord& record) {
  allocator.release(record, device);
}

ATH_RT_SUPPORT_EXPORT void
ath_lock_tensor(Device& device, BackendAllocator& allocator,
                MemoryRecord& record, athena::core::internal::LockType type) {
  allocator.lock(record, device, type);
}

ATH_RT_SUPPORT_EXPORT Event* ath_launch(Device& device,
                                        BackendAllocator& allocator,
                                        LaunchCommand& command, Event* event) {
  return device.launch(allocator, command, event);
}

ATH_RT_SUPPORT_EXPORT uint64_t ath_get_subrecord_addr(MemoryRecord& record,
                                                      uint64_t count,
                                                      uint64_t* shape) {
  return 0; // fixme implement
}

ATH_RT_SUPPORT_EXPORT Device* ath_get_device_for_node(uint64_t nodeId) {
  return nullptr; // fixme implement
}
}