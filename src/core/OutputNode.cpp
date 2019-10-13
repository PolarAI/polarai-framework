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

#include <athena/core/OutputNode.h>

namespace athena::core {
OutputNode::OutputNode(DataType dataType, Context& context, std::string name)
    : AbstractNode({}, dataType, context, std::move(name)) {}
OutputNode::~OutputNode() {
    saveInGraph(false);
}
NodeType OutputNode::getType() const {
    return NodeType::OUTPUT;
}
}  // namespace athena::core
