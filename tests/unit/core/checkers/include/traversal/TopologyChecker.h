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

#include <athena/core/graph/Traversal.h>

#ifndef ATHENA_TOPOLOGYCHECKER_H
#define ATHENA_TOPOLOGYCHECKER_H

namespace athena::tests::unit {
bool checkTopology(const core::Traversal& traversal);
}

#endif // ATHENA_TOPOLOGYCHECKER_H
