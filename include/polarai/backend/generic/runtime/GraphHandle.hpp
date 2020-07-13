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

#include <polarai/backend/generic/runtime/Device.hpp>

#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

namespace polarai {
namespace backend::generic {
class BackendAllocator;
}
namespace core::internal {
class AbstractLoaderInternal;
}
} // namespace polarai

struct GraphHandle {
  std::shared_ptr<polarai::backend::generic::BackendAllocator> allocator;
  std::vector<std::shared_ptr<polarai::backend::generic::Device>> devices;
  std::unordered_map<uint64_t, polarai::core::internal::AbstractLoaderInternal*>
      mLoaders;
  std::set<uint64_t> isHostNode;
};
