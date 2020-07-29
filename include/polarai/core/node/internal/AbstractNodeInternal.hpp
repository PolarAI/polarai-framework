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
#include <polarai/core/Entity.hpp>
#include <polarai/core/graph/EdgeMark.hpp>
#include <polarai/core/node/NodeType.hpp>
#include <polarai/core/tensor/TensorShape.hpp>
#include <polarai/core/tensor/internal/TensorInternal.hpp>
#include <polarai/utils/Index.hpp>
#include <polarai/utils/string/StringView.hpp>

namespace polarai::core::internal {
class POLAR_CORE_EXPORT AbstractNodeInternal : public Entity {
public:
  explicit AbstractNodeInternal(utils::WeakPtr<ContextInternal> context,
                                utils::Index publicNodeIndex,
                                utils::String name = utils::String(""));
  ~AbstractNodeInternal() override;
  void after(const AbstractNodeInternal& node, EdgeMark mark) const;
  void before(const AbstractNodeInternal& node, EdgeMark mark) const;
  [[nodiscard]] virtual NodeType getType() const = 0;
  virtual void clear();
  utils::Allocator<utils::byte> getAllocator();
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
} // namespace polarai::core::internal
