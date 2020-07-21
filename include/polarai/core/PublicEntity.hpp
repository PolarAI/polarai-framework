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

#include <polar_core_export.h>
#include <polarai/core/context/Context.hpp>
#include <polarai/utils/Index.hpp>
#include <polarai/utils/Pointer.hpp>

namespace polarai::core {
namespace internal {
class POLAR_CORE_EXPORT ContextInternal;
}
class POLAR_CORE_EXPORT PublicEntity {
public:
  PublicEntity(utils::SharedPtr<internal::ContextInternal> context,
               utils::Index publicIndex);

  virtual ~PublicEntity() = default;

  Context getContext() const;

  utils::Index getPublicIndex() const;

protected:
  utils::SharedPtr<internal::ContextInternal> mContext;
  utils::Index mPublicIndex;
};
} // namespace polarai::core
