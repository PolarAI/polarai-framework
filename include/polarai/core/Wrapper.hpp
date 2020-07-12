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

#include <polarai/core/ForwardDeclarations.hpp>
#include <polarai/utils/Index.hpp>
#include <polarai/utils/Pointer.hpp>
#include <polar_core_export.h>

#include <iostream>

namespace polarai {
template <typename Type> struct POLAR_CORE_EXPORT Wrapper {
  using PublicType = utils::Index;
};

template <typename Type> struct Returner {
  static typename Wrapper<Type>::PublicType
  returner(utils::SharedPtr<core::internal::ContextInternal> internal,
           utils::Index lastIndex) {
    return lastIndex;
  }
};
} // namespace polarai::core
