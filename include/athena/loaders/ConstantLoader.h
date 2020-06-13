/*
 * Copyright (c) 2019 Athena. All rights reserved.
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

#ifndef ATHENA_CONSTANTLOADER_H
#define ATHENA_CONSTANTLOADER_H

#include <athena/core/loader/AbstractLoader.h>
#include <athena/loaders/internal/ConstantLoaderInternal.h>
#include <athena/core/Wrapper.h>
#include <athena/core/PublicEntity.h>

namespace athena::loaders {
namespace internal {
class ConstantLoaderInternal;
}
class ATH_LOADERS_EXPORT ConstantLoader : public core::PublicEntity {
public:
  using InternalType = internal::ConstantLoaderInternal;

  ConstantLoader(utils::SharedPtr<core::internal::ContextInternal> context,
               utils::Index publicIndex);

  void setConstant(float);

private:
  const internal::ConstantLoaderInternal* internal() const;

  internal::ConstantLoaderInternal* internal();
};
} // namespace athena::loaders

namespace athena {
template <> struct ATH_CORE_EXPORT Wrapper<loaders::ConstantLoader> { using PublicType = loaders::ConstantLoader; };

template <> struct Returner<loaders::ConstantLoader> {
  static typename Wrapper<loaders::ConstantLoader>::PublicType
  returner(utils::SharedPtr<core::internal::ContextInternal> internal,
           utils::Index lastIndex) {
    return loaders::ConstantLoader(std::move(internal), lastIndex);
  }
};
}

#endif // ATHENA_CONSTANTLOADER_H
