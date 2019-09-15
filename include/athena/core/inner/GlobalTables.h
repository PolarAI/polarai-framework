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

#ifndef ATHENA_GLOBALTABLES_H
#define ATHENA_GLOBALTABLES_H

#include <athena/core/inner/Table.h>
#include <athena/core/inner/Tensor.h>

namespace athena::core::inner {
Table<AllocationRecord>& getAllocationTable();
Table<AbstractNode*>& getNodeTable();
Table<Graph*>& getGraphTable();
}  // namespace athena::core::inner

#endif  // ATHENA_GLOBALTABLES_H
