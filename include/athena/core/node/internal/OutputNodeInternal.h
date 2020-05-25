/*
 * Copyright (c) 2019 Athena. All rights reserved.
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

#ifndef ATHENA_OUTPUTNODEINTERNAL_H
#define ATHENA_OUTPUTNODEINTERNAL_H

#include <athena/core/ForwardDeclarations.h>
#include <athena/core/core_export.h>
#include <athena/core/node/internal/AbstractNodeInternal.h>
#include <athena/utils/Pointer.h>

namespace athena::core::internal {
class ATH_CORE_EXPORT OutputNodeInternal : public AbstractNodeInternal {
public:
  OutputNodeInternal() = delete;
  OutputNodeInternal(const OutputNodeInternal& rhs) = default;
  OutputNodeInternal(OutputNodeInternal&& rhs) = default;
  explicit OutputNodeInternal(utils::SharedPtr<ContextInternal> context,
                              utils::Index publicNodeIndex,
                              utils::String name = utils::String(""));
  ~OutputNodeInternal() override;

  OutputNodeInternal& operator=(const OutputNodeInternal& rhs) = delete;
  OutputNodeInternal& operator=(OutputNodeInternal&& rhs) = delete;

  [[nodiscard]] NodeType getType() const override;

  //  template <typename T> Accessor<T> getAccessor() {}
};
} // namespace athena::core::internal

#endif // ATHENA_OUTPUTNODEINTERNAL_H
