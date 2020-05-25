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

#ifndef ATHENA_ENTITY_H
#define ATHENA_ENTITY_H

#include <athena/core/core_export.h>
#include <athena/utils/Index.h>
#include <athena/utils/Pointer.h>
#include <athena/utils/string/String.h>
#include <athena/utils/string/StringView.h>

namespace athena::core {
namespace internal {
class ContextInternal;
}
class ATH_CORE_EXPORT Entity {
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
} // namespace athena::core

#endif // ATHENA_ENTITY_H
