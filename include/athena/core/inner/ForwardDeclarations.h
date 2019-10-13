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

#ifndef ATHENA_FORWARDDECLARATIONS_H
#define ATHENA_FORWARDDECLARATIONS_H

#include <vector>

namespace athena::core {
class AbstractNode;
class Node;
class InputNode;
class OutputNode;
class LossNode;
class Graph;
class Traversal;
class Context;
namespace inner {
struct Cluster;
using Clusters = std::vector<Cluster>;
template <typename Content>
class Table;
class AllocationRecord;
}  // namespace inner
}  // namespace athena::core

#endif  // ATHENA_FORWARDDECLARATIONS_H
