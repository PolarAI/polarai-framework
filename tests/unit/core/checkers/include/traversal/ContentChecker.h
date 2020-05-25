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

#ifndef ATHENA_CONTENTCHECKER_H
#define ATHENA_CONTENTCHECKER_H

#include <athena/core/graph/Traversal.h>
#include <athena/utils/Index.h>

namespace athena::tests::unit {
bool checkTraversalContent(
    const core::Traversal& traversal,
    const std::vector<std::set<athena::utils::Index>>& target);
}

#endif // ATHENA_CONTENTCHECKER_H
