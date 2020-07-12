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
#include <polarai/core/graph/internal/GraphInternal.hpp>
#include <polarai/core/node/AbstractNode.hpp>
#include <polarai/utils/Index.hpp>
#include <polar_core_export.h>

namespace polarai::core {
class POLAR_CORE_EXPORT Graph : public PublicEntity {
public:
  using InternalType = internal::GraphInternal;

  /**
   * Create graph in a context.
   * @param context Reference to context.
   */
  explicit Graph(utils::SharedPtr<internal::ContextInternal> contextInternal,
                 utils::Index publicGraphIndex);

  ~Graph();

  /**
   * Add node to Graph.
   * @param args Arguments for node object creating.
   */
  template <typename TemplateNodeType, typename... Args>
  utils::Index create(Args&&... args) {
    return mContext->getRef<internal::GraphInternal>(mPublicIndex)
        .create<typename TemplateNodeType::InternalType>(
            std::forward<Args>(args)...);
  }

  void connect(utils::Index startNode, utils::Index endNode, EdgeMark edgeMark);

  /**
   *
   * @return Current graph name.
   */
  [[nodiscard]] utils::StringView getName() const;

  std::tuple<Graph, Graph> getGradient(utils::Index targetNodeIndex);

  const Traversal& traverse();

private:
  const internal::GraphInternal* getGraphInternal() const;

  internal::GraphInternal* getGraphInternal();
};
} // namespace polarai::core

namespace polarai {
template <> struct Wrapper<core::Graph> { using PublicType = core::Graph; };

template <> struct Returner<core::Graph> {
  static typename Wrapper<core::Graph>::PublicType
  returner(utils::SharedPtr<core::internal::ContextInternal> internal,
           utils::Index lastIndex) {
    return core::Graph(std::move(internal), lastIndex);
  }
};
}
