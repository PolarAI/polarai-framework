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

#include <polarai/loaders/internal/ConstantLoaderInternal.hpp>

#include <numeric>

namespace polarai::loaders::internal {
ConstantLoaderInternal::ConstantLoaderInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicIndex, float value, utils::String name)
    : core::internal::AbstractLoaderInternal(std::move(context), publicIndex,
                                             std::move(name)),
      mFloatValue(value) {}
void ConstantLoaderInternal::load(core::Accessor<float>& acc) {
  auto& shape = acc.getShape();
  auto total =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  for (uint64_t i = 0; i < total; i++) {
    acc(i) = mFloatValue;
  }
}
//void ConstantLoaderInternal::load(core::Accessor<double>& acc) {
//  auto& shape = acc.getShape();
//  auto total =
//      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
//  for (uint64_t i = 0; i < total; i++) {
//    acc(i) = mDoubleValue;
//  }
//}
void ConstantLoaderInternal::setConstant(float constant) {
  mFloatValue = constant;
}
} // namespace polarai::loaders::internal
