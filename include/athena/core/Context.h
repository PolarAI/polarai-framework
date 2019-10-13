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

#ifndef ATHENA_CONTEXT_H
#define ATHENA_CONTEXT_H

#include <athena/core/inner/ForwardDeclarations.h>
#include <athena/core/inner/InnerFunctions.h>
#include <athena/core/inner/Table.h>

namespace athena::core {
class Context {
    public:
    friend inner::Table<AbstractNode*>& inner::getNodeTable(athena::core::Context& context);
    friend inner::Table<inner::AllocationRecord>& inner::getAllocationTable(athena::core::Context& context);
    friend inner::Table<Graph*>& inner::getGraphTable(athena::core::Context& context);

    private:
    inner::Table<inner::AllocationRecord> allocationTable;
    inner::Table<AbstractNode*> nodeTable;
    inner::Table<Graph*> graphTable;
};
}

#endif  // ATHENA_CONTEXT_H
