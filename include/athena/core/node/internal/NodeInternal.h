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

#ifndef ATHENA_NODEINTERNAL_H
#define ATHENA_NODEINTERNAL_H

#include <athena/core/context/internal/ContextInternal.h>
#include <athena/core/core_export.h>
#include <athena/core/node/internal/AbstractNodeInternal.h>
#include <athena/core/operation/internal/OperationInternal.h>
#include <athena/utils/Index.h>

namespace athena::core::internal {
/**
 * Special type of Node that can not have predecessors
 */
class ATH_CORE_EXPORT NodeInternal : public AbstractNodeInternal {
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
} // namespace athena::core::internal

#endif // ATHENA_INPUTNODEINTERNAL_H
