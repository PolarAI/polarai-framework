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

#ifndef ATHENA_NODETYPE_H
#define ATHENA_NODETYPE_H

#include <athena/core/ForwardDeclarations.h>
#include <athena/core/core_export.h>
#include <athena/utils/error/FatalError.h>
#include <type_traits>

namespace athena::core {
enum class ATH_CORE_EXPORT NodeType {
  UNDEFINED = 0,
  INPUT = 1,
  DEFAULT = 2,
  OUTPUT = 3
};

} // namespace athena::core

#endif // ATHENA_NODETYPE_H
