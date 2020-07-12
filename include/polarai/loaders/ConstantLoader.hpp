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

#include <polarai/core/PublicEntity.hpp>
#include <polarai/core/Wrapper.hpp>
#include <polarai/core/loader/AbstractLoader.hpp>
#include <polarai/loaders/internal/ConstantLoaderInternal.hpp>
#include <polar_loaders_export.h>

namespace polarai::loaders {
namespace internal {
class ConstantLoaderInternal;
}
class POLAR_LOADERS_EXPORT ConstantLoader : public core::PublicEntity {
public:
  using InternalType = internal::ConstantLoaderInternal;

  ConstantLoader(utils::SharedPtr<core::internal::ContextInternal> context,
                 utils::Index publicIndex);

  void setConstant(float);

private:
  const internal::ConstantLoaderInternal* internal() const;

  internal::ConstantLoaderInternal* internal();
};
} // namespace polarai::loaders

namespace polarai {
template <> struct Wrapper<loaders::ConstantLoader> {
  using PublicType = loaders::ConstantLoader;
};

template <> struct Returner<loaders::ConstantLoader> {
  static typename Wrapper<loaders::ConstantLoader>::PublicType
  returner(utils::SharedPtr<core::internal::ContextInternal> internal,
           utils::Index lastIndex) {
    return loaders::ConstantLoader(std::move(internal), lastIndex);
  }
};
} // namespace polarai
