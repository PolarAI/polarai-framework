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

#ifndef ATHENA_TRAVERSAL_H
#define ATHENA_TRAVERSAL_H

#include <athena/core/InputNode.h>
#include <athena/core/Node.h>
#include <athena/core/OutputNode.h>
#include <athena/core/core_export.h>
#include <athena/core/inner/TupleContainers.h>

namespace athena::core {
namespace inner {
struct ATH_CORE_EXPORT Dependency {
    size_t nodeIndex;
    size_t mark;
    Dependency(size_t nodeIndex, size_t mark)
        : nodeIndex(nodeIndex), mark(mark) {}
};

template <typename TemplateNodeType>
struct ATH_CORE_EXPORT NodeDependencies {
    size_t nodeIndex;
    std::vector<Dependency> input;
    std::vector<Dependency> output;
    NodeDependencies(size_t nodeIndex,
                     std::vector<Dependency> input,
                     std::vector<Dependency> output)
        : nodeIndex(nodeIndex),
          input(std::move(input)),
          output(std::move(output)) {}
};

struct ATH_CORE_EXPORT NodeState {
    size_t inputCount;
    std::vector<Dependency> input;
    std::vector<Dependency> output;
};

struct ATH_CORE_EXPORT Cluster {
    using Content = inner::TupleContainersWithWrappers<std::vector,
                                                       inner::NodeDependencies,
                                                       Node,
                                                       InputNode,
                                                       OutputNode,
                                                       LossNode>::Holder;
    size_t nodeCount;
    Content content;
    template <typename TemplateNodeType>
    std::vector<inner::NodeDependencies<TemplateNodeType>> &get() {
        return std::get<std::vector<inner::NodeDependencies<TemplateNodeType>>>(
            content);
    }
    template <typename TemplateNodeType>
    const std::vector<inner::NodeDependencies<TemplateNodeType>> &get() const {
        return std::get<std::vector<inner::NodeDependencies<TemplateNodeType>>>(
            content);
    }
};
}  // namespace inner

/**
 * Graph traversal
 */
class ATH_CORE_EXPORT Traversal {
    private:
    inner::Clusters clusters;
    bool mIsValidTraversal;

    public:
    const inner::Clusters &getClusters() const {
        return clusters;
    }
    bool isValidTraversal() const {
        return mIsValidTraversal;
    }
    friend inner::Clusters &inner::getClusters(Traversal &traversal);
    friend void inner::setTraversalValidity(Traversal &traversal, bool flag);
};
}  // namespace athena::core

#endif  // ATHENA_TRAVERSAL_H
