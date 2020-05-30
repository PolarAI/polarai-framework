#include <memory>

#pragma once

namespace athena::backend::llvm {
  class BackendAllocator;
}

struct GraphHandle {
  std::shared_ptr<athena::backend::llvm::BackendAllocator> allocator;
};
