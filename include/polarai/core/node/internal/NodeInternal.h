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

#include <polar_core_export.h>
#include <polarai/core/context/internal/ContextInternal.hpp>
#include <polarai/core/node/internal/AbstractNodeInternal.hpp>
#include <polarai/core/operation/internal/OperationInternal.hpp>
#include <polarai/utils/Index.hpp>

namespace polarai::core::internal {
/**
 * Special type of Node that can not have predecessors
 */
class POLAR_CORE_EXPORT NodeInternal : public AbstractNodeInternal {
public:
  NodeInternal(utils::SharedPtr<ContextInternal> context,
               utils::Index publicNodeIndex, utils::Index operationIndex,
               utils::String name = utils::String(""));

  [[nodiscard]] NodeType getType() const override;

  OperationInternal* operationPtr();

  [[nodiscard]] const OperationInternal* getOperationPtr() const;

private:
  utils::Index mOperationIndex;
};
} // namespace polarai::core::internal
