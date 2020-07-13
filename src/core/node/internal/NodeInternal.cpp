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

#include <polarai/core/node/internal/NodeInternal.hpp>

#include <iostream>

namespace polarai::core::internal {
NodeInternal::NodeInternal(utils::SharedPtr<ContextInternal> context,
                           utils::Index publicNodeIndex,
                           utils::Index operationIndex, utils::String name)
    : AbstractNodeInternal(std::move(context), publicNodeIndex,
                           std::move(name)),
      mOperationIndex(operationIndex) {}

NodeType NodeInternal::getType() const { return NodeType::DEFAULT; }

OperationInternal* NodeInternal::operationPtr() {
  return mContext.lock()->getPtr<OperationInternal>(mOperationIndex);
}

const OperationInternal* NodeInternal::getOperationPtr() const {
  return mContext.lock()->getPtr<OperationInternal>(mOperationIndex);
}
} // namespace polarai::core::internal
