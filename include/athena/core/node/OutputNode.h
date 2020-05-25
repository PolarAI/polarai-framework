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

#ifndef ATHENA_OUTPUTNODE_H
#define ATHENA_OUTPUTNODE_H

#include <athena/core/core_export.h>
#include <athena/core/node/AbstractNode.h>
#include <athena/core/node/internal/OutputNodeInternal.h>

namespace athena::core {
namespace internal {
class OutputNodeInternal;
}

/**
 * A Node represents a piece of data loading to graph.
 */
class ATH_CORE_EXPORT OutputNode : public AbstractNode {
public:
  using InternalType = internal::OutputNodeInternal;
};
} // namespace athena::core

#endif // ATHENA_OUTPUTNODE_H
