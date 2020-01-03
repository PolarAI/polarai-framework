/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_DOTMODEL_H
#define ATHENA_DOTMODEL_H

#include <athena/core/Graph.h>
#include <athena/model/model_export.h>

#include <ostream>

namespace athena::model {
/**
 * Print graph to DOT format for debug purposes
 */
class ATH_MODEL_EXPORT DotModel {
public:
  static void exportGraph(core::Graph& graph, std::ostream& stream);
};
} // namespace athena::model

#endif // ATHENA_DOTMODEL_H
