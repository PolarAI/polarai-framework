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

#include <polarai/core/loader/AbstractLoader.hpp>
#include <polarai/loaders/internal/MemcpyLoaderInternal.hpp>
#include <polarai/core/Wrapper.hpp>
#include <polarai/core/PublicEntity.hpp>
#include <polar_loaders_export.h>

namespace polarai::loaders {
namespace internal {
class MemcpyLoaderInternal;
}
class POLAR_LOADERS_EXPORT MemcpyLoader : public core::PublicEntity {
public:
  using InternalType = internal::MemcpyLoaderInternal;

  MemcpyLoader(utils::SharedPtr<core::internal::ContextInternal> context,
    utils::Index publicIndex);

  void setPointer(void* source, size_t size);

private:
  const internal::MemcpyLoaderInternal* internal() const;

  internal::MemcpyLoaderInternal* internal();
};
} // namespace polarai::loaders

namespace polarai {
template <> struct Wrapper<loaders::MemcpyLoader> { using PublicType = loaders::MemcpyLoader; };

template <> struct Returner<loaders::MemcpyLoader> {
  static typename Wrapper<loaders::MemcpyLoader>::PublicType
  returner(utils::SharedPtr<core::internal::ContextInternal> internal,
           utils::Index lastIndex) {
    return loaders::MemcpyLoader(std::move(internal), lastIndex);
  }
};
}
