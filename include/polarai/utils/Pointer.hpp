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

#include <polarai/utils/Defines.hpp>

#include <memory>

namespace polarai::utils {
template <typename Type> using UniquePtr = std::unique_ptr<Type>;

template <typename Type, typename... Args>
ATH_FORCE_INLINE UniquePtr<Type> makeUnique(Args&&... args) {
  return std::make_unique<Type>(std::forward<Args>(args)...);
}

template <typename Type> using SharedPtr = std::shared_ptr<Type>;

template <typename Type, typename... Args>
ATH_FORCE_INLINE SharedPtr<Type> makeShared(Args&&... args) {
  return std::make_shared<Type>(std::forward<Args>(args)...);
}

template <typename Type> using WeakPtr = std::weak_ptr<Type>;

template <typename Type> void SharedDeleter(Type* ptr) {}
} // namespace polarai::utils
