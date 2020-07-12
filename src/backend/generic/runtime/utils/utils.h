#pragma once
#include <athena/backend/llvm/BackendAllocator.h>
#include <athena/backend/llvm/runtime/TensorInfo.h>

inline auto tensorInfoToRecord(TensorInfo* tensor)
    -> athena::backend::llvm::MemoryRecord {
  athena::backend::llvm::MemoryRecord record;
  record.virtualAddress = tensor->virtAddr;
  record.allocationSize = athena::core::sizeOfDataType(
      static_cast<athena::core::DataType>(tensor->dataType));
  for (int i = 0; i < tensor->dims; i++) {
    record.allocationSize *= tensor->shape[i];
  }
  return record;
}

inline auto tensorSize(TensorInfo* tensor) {
  uint64_t total = 1;
  for (int i = 0; i < tensor->dims; i++) {
    total *= tensor->shape[i];
  }
  return total;
}
