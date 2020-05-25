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

#ifndef ATHENA_GRAPH_H
#define ATHENA_GRAPH_H

#include <athena/core/context/Context.h>
#include <athena/core/core_export.h>
#include <athena/core/graph/internal/GraphInternal.h>
#include <athena/core/node/AbstractNode.h>
#include <athena/utils/Index.h>

namespace athena::core {
class ATH_CORE_EXPORT Graph : public PublicEntity {
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

  std::tuple<Graph, Graph> getGradient();

  const Traversal& traverse();

private:
  const internal::GraphInternal* getGraphInternal() const;

  internal::GraphInternal* getGraphInternal();
};

template <> struct ATH_CORE_EXPORT Wrapper<Graph> { using PublicType = Graph; };

template <> struct Returner<Graph> {
  static typename Wrapper<Graph>::PublicType
  returner(utils::SharedPtr<internal::ContextInternal> internal,
           utils::Index lastIndex) {
    return Graph(std::move(internal), lastIndex);
  }
};
} // namespace athena::core

#endif // ATHENA_GRAPH_H
