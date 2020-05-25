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

#include <athena/core/operation/internal/OperationInternal.h>

namespace athena::core::internal {
OperationInternal::OperationInternal(utils::WeakPtr<ContextInternal> context,
                                     utils::Index publicNodeIndex,
                                     utils::String name)
    : Entity(std::move(context), publicNodeIndex, std::move(name)) {}
} // namespace athena::core::internal
