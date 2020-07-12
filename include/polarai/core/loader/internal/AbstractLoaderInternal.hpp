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
#include <polarai/core/context/internal/ContextInternal.hpp>
#include <polarai/core/loader/internal/TensorAllocator.hpp>
#include <polarai/core/tensor/Accessor.hpp>
#include <polarai/utils/string/StringView.hpp>

namespace polarai::core::internal {
/**
 * Loaders is a concept that helps Athena put user data into Graph
 */
class POLAR_CORE_EXPORT AbstractLoaderInternal : public Entity {
public:
  AbstractLoaderInternal(utils::WeakPtr<ContextInternal> context,
                         utils::Index publicIndex,
                         utils::String name = utils::String(""));

  virtual void load(Accessor<float>&) = 0;
  // virtual void load(Accessor<double>&) = 0;

protected:
  utils::String mName;
};
} // namespace polarai::core::internal
