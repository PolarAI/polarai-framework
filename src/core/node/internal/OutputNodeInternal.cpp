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

#include <polarai/core/node/internal/OutputNodeInternal.hpp>

namespace polarai::core::internal {
OutputNodeInternal::OutputNodeInternal(
    utils::SharedPtr<ContextInternal> context, utils::Index publicNodeIndex,
    utils::String name)
    : AbstractNodeInternal(std::move(context), publicNodeIndex,
                           std::move(name)) {}
OutputNodeInternal::~OutputNodeInternal() {}
NodeType OutputNodeInternal::getType() const { return NodeType::OUTPUT; }
} // namespace polarai::core::internal
