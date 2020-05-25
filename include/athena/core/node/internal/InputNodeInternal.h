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

#ifndef ATHENA_INPUTNODEINTERNAL_H
#define ATHENA_INPUTNODEINTERNAL_H

#include <athena/core/context/Context.h>
#include <athena/core/core_export.h>
#include <athena/core/loader/internal/AbstractLoaderInternal.h>
#include <athena/core/node/internal/AbstractNodeInternal.h>
#include <athena/core/tensor/DataType.h>

namespace athena::core::internal {
/**
 * Special type of Node that can not have predecessors
 */
class ATH_CORE_EXPORT InputNodeInternal : public AbstractNodeInternal {
public:
  InputNodeInternal(utils::SharedPtr<ContextInternal> context,
                    utils::Index publicNodeIndex, TensorShape tensorShape,
                    DataType dataType, bool isFrozen, utils::Index loaderIndex,
                    utils::String name = utils::String(""));

  [[nodiscard]] NodeType getType() const override;

  bool isFrozen() const;

protected:
  bool mIsFrozen;
  utils::Index mLoaderIndex;
};
} // namespace athena::core::internal

#endif // ATHENA_INPUTNODEINTERNAL_H
