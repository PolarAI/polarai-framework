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

#ifndef ATHENA_INPUTNODE_H
#define ATHENA_INPUTNODE_H

#include <athena/core/AbstractLoader.h>
#include <athena/core/AbstractNode.h>

namespace athena::core {
/**
 * Special type of Node that can not have predecessors
 */
class InputNode : public AbstractNode {
    protected:
    AbstractLoader* mLoader;

    public:
    InputNode() = delete;
    InputNode(const InputNode& rhs) = default;
    InputNode(InputNode&& rhs) noexcept;
    InputNode(TensorShape shape,
              DataType dataType,
              AbstractLoader& loader,
              std::string name = "");
    ~InputNode() override;

    InputNode& operator=(const InputNode& rhs) = default;
    InputNode& operator=(InputNode&& rhs) noexcept;

    NodeType getType() const override;
    const AbstractLoader& getLoader() const;
    const AbstractLoader* getLoaderPtr() const;
    AbstractLoader& loader();
    const AbstractLoader& loader() const;
    void setLoader(AbstractLoader& loader);
    void clear() override;
};
}  // namespace athena::core

#endif  // ATHENA_INPUTNODE_H
