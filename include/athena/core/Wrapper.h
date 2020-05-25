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

#ifndef ATHENA_WRAPPER_H
#define ATHENA_WRAPPER_H

#include <athena/core/ForwardDeclarations.h>
#include <athena/core/core_export.h>
#include <athena/utils/Index.h>
#include <athena/utils/Pointer.h>
#include <iostream>

namespace athena::core {
template <typename Type> struct ATH_CORE_EXPORT Wrapper {
  using PublicType = utils::Index;
};

template <typename Type> struct Returner {
  static typename Wrapper<Type>::PublicType
  returner(utils::SharedPtr<internal::ContextInternal> internal,
           utils::Index lastIndex) {
    return lastIndex;
  }
};
} // namespace athena::core
#endif // ATHENA_WRAPPER_H
