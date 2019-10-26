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

#include <athena/core/InputNode.h>

namespace athena::core {
InputNode::InputNode(InputNode&& rhs) noexcept
    : AbstractNode(std::move(rhs)), mLoader(rhs.mLoader) {
    rhs.mLoader = nullptr;
}
InputNode::InputNode(TensorShape shape,
                     DataType dataType,
                     AbstractLoader& loader,
                     Context& context,
                     bool isFrozen,
                     std::string name)
    : AbstractNode(std::move(shape), dataType, context, std::move(name)),
      mIsFrozen(isFrozen),
      mLoader(&loader) {}
InputNode::~InputNode() {
    saveInGraph(false);
}
NodeType InputNode::getType() const {
    return NodeType::INPUT;
}
const AbstractLoader& InputNode::getLoader() const {
    return *mLoader;
}
const AbstractLoader* InputNode::getLoaderPtr() const {
    return mLoader;
}
AbstractLoader& InputNode::loader() {
    return *mLoader;
}
const AbstractLoader& InputNode::loader() const {
    return *mLoader;
}
void InputNode::setLoader(AbstractLoader& loader) {
    mLoader = &loader;
}
void InputNode::clear() {
    mLoader = nullptr;
    AbstractNode::clear();
}
bool InputNode::isFrozen() {
    return mIsFrozen;
}
}  // namespace athena::core