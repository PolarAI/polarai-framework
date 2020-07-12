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
#include <polarai/utils/Index.hpp>
#include <polarai/utils/Pointer.hpp>
#include <polarai/utils/string/String.hpp>
#include <polarai/utils/string/StringView.hpp>

namespace polarai::core {
namespace internal {
class ContextInternal;
}
class POLAR_CORE_EXPORT Entity {
public:
  Entity(utils::WeakPtr<internal::ContextInternal> context,
         utils::Index publicIndex, utils::String name = "");

  virtual ~Entity() = default;

  utils::SharedPtr<internal::ContextInternal> getContext() const;

  utils::Index getPublicIndex() const;

  utils::StringView getName() const;

protected:
  utils::WeakPtr<internal::ContextInternal> mContext;
  utils::Index mPublicIndex;
  utils::String mName;
};
} // namespace polarai::core
