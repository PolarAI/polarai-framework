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

#ifndef ATHENA_PUBLICENTITY_H
#define ATHENA_PUBLICENTITY_H

#include <athena/core/context/Context.h>
#include <athena/core/core_export.h>
#include <athena/utils/Index.h>
#include <athena/utils/Pointer.h>

namespace athena::core {
namespace internal {
class ContextInternal;
}
class ATH_CORE_EXPORT PublicEntity {
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
} // namespace athena::core

#endif // ATHENA_PUBLICENTITY_H
