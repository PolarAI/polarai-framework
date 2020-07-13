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

#include <polarai/core/node/internal/InputNodeInternal.hpp>

namespace polarai::core::internal {
InputNodeInternal::InputNodeInternal(utils::SharedPtr<ContextInternal> context,
                                     utils::Index publicNodeIndex,
                                     TensorShape tensorShape, DataType dataType,
                                     bool isFrozen, utils::Index loaderIndex,
                                     utils::String name)
    : AbstractNodeInternal(context, publicNodeIndex,
                           context->create<TensorInternal>(
                               context, context->getNextPublicIndex(), dataType,
                               std::move(tensorShape)),
                           std::move(name)),
      mIsFrozen(isFrozen), mLoaderIndex(loaderIndex) {}

NodeType InputNodeInternal::getType() const { return NodeType::INPUT; }

bool InputNodeInternal::isFrozen() const { return mIsFrozen; }
} // namespace polarai::core::internal
