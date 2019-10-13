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

#include <athena/core/Context.h>

namespace athena::core::inner {
inner::Table<AllocationRecord>& getAllocationTable(athena::core::Context& context) {
    return context.allocationTable;
}
inner::Table<Graph*>& getGraphTable(athena::core::Context& context) {
    return context.graphTable;
}
inner::Table<AbstractNode*>& getNodeTable(athena::core::Context& context) {
    return context.nodeTable;
}
}