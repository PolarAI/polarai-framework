
//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#ifndef ATHENA_MEMCPYLOADER_H
#define ATHENA_MEMCPYLOADER_H

#include <athena/core/loader/AbstractLoader.h>
#include <athena/loaders/internal/MemcpyLoaderInternal.h>
#include <athena/core/Wrapper.h>
#include <athena/core/PublicEntity.h>


namespace athena::loaders {
namespace internal {
class MemcpyLoaderInternal;
}
class ATH_LOADERS_EXPORT MemcpyLoader : public core::PublicEntity {
public:
  using InternalType = internal::MemcpyLoaderInternal;

  MemcpyLoader(utils::SharedPtr<core::internal::ContextInternal> context,
    utils::Index publicIndex);

  void setPointer(void* source, size_t size);

private:
  const internal::MemcpyLoaderInternal* internal() const;

  internal::MemcpyLoaderInternal* internal();
};
} // namespace athena::loaders

namespace athena {
template <> struct ATH_CORE_EXPORT Wrapper<loaders::MemcpyLoader> { using PublicType = loaders::MemcpyLoader; };

template <> struct Returner<loaders::MemcpyLoader> {
  static typename Wrapper<loaders::MemcpyLoader>::PublicType
  returner(utils::SharedPtr<core::internal::ContextInternal> internal,
           utils::Index lastIndex) {
    return loaders::MemcpyLoader(std::move(internal), lastIndex);
  }
};
}

#endif // ATHENA_MEMCPYLOADER_H
