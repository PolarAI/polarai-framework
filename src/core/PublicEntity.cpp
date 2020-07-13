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

#include <polarai/core/PublicEntity.hpp>
#include <polarai/core/context/Context.hpp>

namespace polarai::core {
PublicEntity::PublicEntity(utils::SharedPtr<internal::ContextInternal> context,
                           utils::Index publicIndex)
    : mContext(std::move(context)), mPublicIndex(publicIndex) {}

Context PublicEntity::getContext() const { return Context(mContext); }

utils::Index PublicEntity::getPublicIndex() const { return mPublicIndex; }
} // namespace polarai::core
