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
#include <polarai/core/loader/internal/TensorAllocator.hpp>
#include <polarai/core/node/AbstractNode.hpp>
#include <polarai/core/node/OutputNodeAccessor.hpp>
#include <polarai/core/node/internal/OutputNodeInternal.hpp>

namespace polarai::core {
namespace internal {
class OutputNodeInternal;
}

/**
 * A Node represents a piece of data loading to graph.
 */
class POLAR_CORE_EXPORT OutputNode : public AbstractNode {
public:
  using InternalType = internal::OutputNodeInternal;

  template <typename T>
  OutputNodeAccessor<T> getAccess(internal::TensorAllocator& allocator) {
    auto tensorIdx =
        mContext->getRef<InternalType>(mPublicIndex).getTensorIndex();
    auto& tensorRef = mContext->getRef<internal::TensorInternal>(tensorIdx);
    return OutputNodeAccessor<T>(allocator, tensorRef);
  }
};
} // namespace polarai::core
