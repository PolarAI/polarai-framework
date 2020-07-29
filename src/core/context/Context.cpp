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

#include <polarai/core/context/Context.hpp>
#include <polarai/core/context/internal/ContextInternal.hpp>

namespace polarai::core {
Context::Context(utils::Allocator<utils::byte> allocator,
                 size_t defaultCapacity, size_t elementAverageSize)
    : mInternal(utils::makeShared<internal::ContextInternal>(
          std::move(allocator), defaultCapacity, elementAverageSize)) {}

Context::Context(utils::SharedPtr<internal::ContextInternal> ptr)
    : mInternal(std::move(ptr)) {}

Context::~Context() {}

utils::SharedPtr<internal::ContextInternal> Context::internal() {
  return mInternal;
}

utils::SharedPtr<const internal::ContextInternal> Context::internal() const {
  return mInternal;
}

utils::Allocator<utils::byte>& Context::getAllocator() {
  return mInternal->getAllocator();
}
} // namespace polarai::core
