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

#ifndef ATHENA_ABSTRACTNODEINTERNAL_H
#define ATHENA_ABSTRACTNODEINTERNAL_H

#include <athena/core/Entity.h>
#include <athena/core/core_export.h>
#include <athena/core/graph/EdgeMark.h>
#include <athena/core/node/NodeType.h>
#include <athena/core/tensor/TensorShape.h>
#include <athena/core/tensor/internal/TensorInternal.h>
#include <athena/utils/Index.h>
#include <athena/utils/string/StringView.h>

namespace athena::core::internal {
class ATH_CORE_EXPORT AbstractNodeInternal : public Entity {
public:
  explicit AbstractNodeInternal(utils::WeakPtr<ContextInternal> context,
                                utils::Index publicNodeIndex,
                                utils::String name = utils::String(""));
  ~AbstractNodeInternal() override;
  void after(const AbstractNodeInternal& node, EdgeMark mark) const;
  void before(const AbstractNodeInternal& node, EdgeMark mark) const;
  [[nodiscard]] virtual NodeType getType() const = 0;
  virtual void clear();
  utils::Allocator getAllocator();
  [[nodiscard]] const TensorInternal* getTensorPtr() const;
  TensorInternal* getTensorPtr();
  utils::Index getTensorIndex() const;
  void setTensorIndex(utils::Index tensorIndex);

protected:
  explicit AbstractNodeInternal(utils::WeakPtr<ContextInternal> context,
                                utils::Index publicNodeIndex,
                                utils::Index tensorIndex,
                                utils::String name = utils::String(""));
  utils::Index mTensorIndex;
};
} // namespace athena::core::internal

#endif // ATHENA_ABSTRACTNODEINTERNAL_H
