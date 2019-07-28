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
#include <athena/core/inner/GlobalTables.h>
#include <athena/core/inner/Tensor.h>

#include <string>
#include <utility>
#include <vector>

namespace athena::core {
/**
 * Operation is an abstract computation, like addition or multiplication
 */
class Operation {
    protected:
    std::string mName;

    public:
    explicit Operation(std::string&& name) : mName(std::move(name)){};
    virtual inner::Tensor& getResultTensor(
        std::vector<inner::Tensor*> args) const = 0;
    virtual inner::Tensor& getDerivativeTensor(std::vector<inner::Tensor*> args,
                                               int argNo) const = 0;
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
    virtual void genDerivative(AbstractGenerator& g,
                               std::vector<inner::Tensor*>& operationArguments,
                               int argNo) const = 0;
    /**
     *
     * @return Name of Operation
     */
    std::string getName() const;

    virtual size_t getOperandsCount() const = 0;
};

class OperationDummy : public Operation {
    public:
    explicit OperationDummy(std::string name) : Operation(std::move(name)){};

    inner::Tensor& getResultTensor(
        std::vector<inner::Tensor*> args) const override {
        new FatalError(1, "NOT IMPL");
        return *inner::getTensorRegistry()[0];
    }

    inner::Tensor& getDerivativeTensor(std::vector<inner::Tensor*> args,
                                       int argNo) const override {
        new FatalError(1, "NOT IMPL");
        return *inner::getTensorRegistry()[0];
    }

    void gen(AbstractGenerator& g,
             std::vector<inner::Tensor*>& operationArguments) const override {
        new FatalError(1, "NOT IMPL");
    }

    void genDerivative(AbstractGenerator& g,
                       std::vector<inner::Tensor*>& operationArguments,
                       int argNo) const override {
        new FatalError(1, "NOT IMPL");
    }
};
}  // namespace athena::core

#endif  // ATHENA_OPERATION_H
