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

#ifndef ATHENA_NODE_H
#define ATHENA_NODE_H

#include <athena/core/node/AbstractNode.h>
#include <athena/core/node/internal/NodeInternal.h>

namespace athena::core {
/**
 * A Node represents a piece of data loading to graph.
 */
class ATH_CORE_EXPORT Node {
public:
  using InternalType = internal::NodeInternal;
};
} // namespace athena::core

#endif // ATHENA_NODE_H
