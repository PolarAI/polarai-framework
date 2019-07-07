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

#include <athena/core/inner/GlobalTables.h>

namespace athena::core::inner {
Table<AllocationRecord>& getAllocationTable() {
    static Table<AllocationRecord> table;
    return table;
}

Table<AbstractNode*>& getNodeTable() {
    static Table<AbstractNode*> table;
    return table;
}
Table<Graph*>& getGraphTable() {
    static Table<Graph*> table;
    return table;
}
}  // namespace athena::core::inner