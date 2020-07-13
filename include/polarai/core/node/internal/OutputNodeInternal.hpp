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
#include <polarai/core/ForwardDeclarations.hpp>
#include <polarai/core/node/internal/AbstractNodeInternal.hpp>
#include <polarai/utils/Pointer.hpp>

namespace polarai::core::internal {
class POLAR_CORE_EXPORT OutputNodeInternal : public AbstractNodeInternal {
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
};
} // namespace polarai::core::internal
