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

#include <polarai/loaders/internal/DummyLoaderInternal.hpp>

namespace polarai::loaders::internal {
DummyLoaderInternal::DummyLoaderInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicIndex, utils::String name)
    : core::internal::AbstractLoaderInternal(std::move(context), publicIndex,
                                             std::move(name)) {}
void DummyLoaderInternal::load(core::Accessor<float>& acc) {}
//void DummyLoaderInternal::load(core::Accessor<double>& acc) {}
} // namespace polarai::loaders::internal
