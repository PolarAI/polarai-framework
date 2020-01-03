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

#ifndef ATHENA_NATIVEMODEL_H
#define ATHENA_NATIVEMODEL_H

#include <athena/core/Graph.h>
#include <athena/model/model_export.h>

#include <istream>
#include <ostream>

namespace athena::model {
class ATH_MODEL_EXPORT NativeModel {
public:
  static void serializeGraph(core::Graph& graph, std::ostream& stream);
  static void deserializeGraph(core::Graph& graph, std::istream& stream);
  static void saveGraphToFile(core::Graph& graph, const std::string& filename);
  static void readGraphFromFile(core::Graph& graph, const std::string& name);
};
} // namespace athena::model

#endif // ATHENA_NATIVEMODEL_H
