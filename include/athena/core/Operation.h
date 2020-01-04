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

#ifndef ATHENA_OPERATION_H
#define ATHENA_OPERATION_H

#include <athena/core/AbstractGenerator.h>
#include <athena/core/FatalError.h>
#include <athena/core/core_export.h>
#include <athena/core/inner/InnerFunctions.h>
#include <athena/core/inner/Tensor.h>

#include <string>
#include <utility>
#include <vector>

namespace athena::core {
/**
 * Operation is an abstract computation, like addition or multiplication
 */
class ATH_CORE_EXPORT Operation {
    protected:
    std::string mName;

    public:
    explicit Operation(std::string name) : mName(std::move(name)){};

    virtual std::shared_ptr<core::inner::Tensor> createTensor(
        core::Context& context, std::vector<core::inner::Tensor *> args) const = 0;

    /**
     * Generate code for Operation
     * @param g Generator to be used
     * @param operationArguments Necessary arguments specific
     * to Generator implementation
     */
    virtual void gen(AbstractGenerator& g,
                     std::vector<inner::Tensor*>& operationArguments) const = 0;

    /**
     * Generate code for Operation derivative
     * @param g Generator to be used
     * @param operationArguments Necessary arguments specific
     * to Generator implementation
     * @param argNo Index of argument that derivative will be computed to
     */
    virtual void genIncomingDerivative(
        core::AbstractGenerator &g,
        std::vector<core::inner::Tensor *> &operationArguments,
        core::inner::Tensor &derivativeOfIncomingNode,
        core::inner::Tensor &ownDerivative,
        size_t argumentMark) const = 0;

    virtual void genOwnDerivative(AbstractGenerator& g,
                                  std::map<size_t, std::shared_ptr<inner::Tensor>> &outgoingDerivatives,
                                  inner::Tensor &ownDerivative) const;

    /**
     *
     * @return Name of Operation
     */
    std::string getName() const;

    virtual size_t getOperandsCount() const = 0;

    virtual std::string serialize() const = 0;

    static Operation* deserialize(const std::string& data) {
        return nullptr;
    };
};
}  // namespace athena::core

#endif  // ATHENA_OPERATION_H
