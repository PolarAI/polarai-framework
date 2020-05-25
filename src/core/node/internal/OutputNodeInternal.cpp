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

#include <athena/core/node/internal/OutputNodeInternal.h>

namespace athena::core::internal {
OutputNodeInternal::OutputNodeInternal(
    utils::SharedPtr<ContextInternal> context, utils::Index publicNodeIndex,
    utils::String name)
    : AbstractNodeInternal(std::move(context), publicNodeIndex,
                           std::move(name)) {}
OutputNodeInternal::~OutputNodeInternal() {}
NodeType OutputNodeInternal::getType() const { return NodeType::OUTPUT; }
} // namespace athena::core::internal
