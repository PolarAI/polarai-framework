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

#include <polarai/core/context/Context.hpp>
#include <polarai/core/loader/internal/AbstractLoaderInternal.hpp>
#include <polarai/core/node/internal/AbstractNodeInternal.hpp>
#include <polarai/core/tensor/DataType.hpp>
#include <polar_core_export.h>

namespace polarai::core::internal {
/**
 * Special type of Node that can not have predecessors
 */
class POLAR_CORE_EXPORT InputNodeInternal : public AbstractNodeInternal {
public:
  InputNodeInternal(utils::SharedPtr<ContextInternal> context,
                    utils::Index publicNodeIndex, TensorShape tensorShape,
                    DataType dataType, bool isFrozen, utils::Index loaderIndex,
                    utils::String name = utils::String(""));

  [[nodiscard]] NodeType getType() const override;

  bool isFrozen() const;

  utils::Index getLoader() { return mLoaderIndex; }

protected:
  bool mIsFrozen;
  utils::Index mLoaderIndex;
};
} // namespace polarai::core::internal
