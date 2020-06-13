#include <athena/backend/llvm/runtime/Device.h>

#include <memory>
#include <vector>
#include <set>

#pragma once

namespace athena {
namespace backend::llvm {
class BackendAllocator;
}
namespace core::internal {
class AbstractLoaderInternal;
}
} // namespace athena

struct GraphHandle {
  std::shared_ptr<athena::backend::llvm::BackendAllocator> allocator;
  std::vector<std::shared_ptr<athena::backend::llvm::Device>> devices;
  std::unordered_map<uint64_t, athena::core::internal::AbstractLoaderInternal*>
      mLoaders;
  std::set<uint64_t> isHostNode;
};
