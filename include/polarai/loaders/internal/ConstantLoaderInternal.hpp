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

#include <polarai/core/loader/internal/AbstractLoaderInternal.hpp>
#include <polar_loaders_export.h>

namespace polarai::loaders::internal {
class POLAR_LOADERS_EXPORT ConstantLoaderInternal
    : public core::internal::AbstractLoaderInternal {
public:
  ConstantLoaderInternal(
      utils::WeakPtr<core::internal::ContextInternal> context,
      utils::Index publicIndex, float value,
      utils::String name = utils::String(""));

  void load(core::Accessor<float>&) override;
  // void load(core::Accessor<double>&) override;

  void setConstant(float constant);

protected:
  float mFloatValue;
};
} // namespace polarai::loaders::internal
