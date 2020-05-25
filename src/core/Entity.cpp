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

#include <athena/core/Entity.h>

namespace athena::core {
Entity::Entity(utils::WeakPtr<internal::ContextInternal> context,
               utils::Index publicIndex, utils::String name)
    : mContext(std::move(context)), mPublicIndex(publicIndex),
      mName(std::move(name)) {}

utils::SharedPtr<internal::ContextInternal> Entity::getContext() const {
  return mContext.lock();
}

utils::Index Entity::getPublicIndex() const { return mPublicIndex; }

utils::StringView Entity::getName() const { return utils::StringView(mName); }
} // namespace athena::core