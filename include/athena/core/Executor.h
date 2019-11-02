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

#ifndef ATHENA_EXECUTOR_H
#define ATHENA_EXECUTOR_H

#include <athena/core/Graph.h>
#include <athena/core/core_export.h>

namespace athena::core {

class ATH_CORE_EXPORT Executor {
    public:
    virtual void setGraph(Graph &graph) = 0;
    virtual void evaluate() = 0;
    virtual void optimizeGraph() = 0;
};

}  // namespace athena::core
#endif  // ATHENA_EXECUTOR_H
