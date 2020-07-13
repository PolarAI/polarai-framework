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

#include <polarai/core/ForwardDeclarations.hpp>
#include <polarai/core/Wrapper.hpp>
#include <polarai/core/context/internal/ContextInternal.hpp>
#include <polarai/utils/Index.hpp>
#include <polarai/utils/allocator/Allocator.hpp>
#include <polar_core_export.h>

namespace polarai::core {
class POLAR_CORE_EXPORT Context {
public:
  explicit Context(utils::Allocator allocator = utils::Allocator(),
                   size_t defaultCapacity = 100,
                   size_t elementAverageSize = 32);

  explicit Context(utils::SharedPtr<internal::ContextInternal> ptr);

  ~Context();

  template <typename Type, typename... Args>
  typename Wrapper<Type>::PublicType create(Args&&... args);

  utils::Allocator& getAllocator();

  utils::SharedPtr<internal::ContextInternal> internal();

  [[nodiscard]] utils::SharedPtr<const internal::ContextInternal>
  internal() const;

private:
  utils::SharedPtr<internal::ContextInternal> mInternal;
};

template <typename Type, typename... Args>
typename Wrapper<Type>::PublicType Context::create(Args&&... args) {
  auto index = mInternal->create<typename Type::InternalType>(
      mInternal, mInternal->getNextPublicIndex(), std::forward<Args>(args)...);
  return Returner<Type>::returner(mInternal, index);
}
} // namespace polarai::core
