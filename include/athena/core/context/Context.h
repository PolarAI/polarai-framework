/*
 * Copyright (c) 2018 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_CONTEXT_H
#define ATHENA_CONTEXT_H

#include <athena/core/ForwardDeclarations.h>
#include <athena/core/Wrapper.h>
#include <athena/core/context/internal/ContextInternal.h>
#include <athena/core/core_export.h>
#include <athena/utils/Index.h>
#include <athena/utils/allocator/Allocator.h>

namespace athena::core {
class ATH_CORE_EXPORT Context {
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
} // namespace athena::core

#endif // ATHENA_CONTEXT_H
