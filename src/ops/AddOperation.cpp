/*
 * Copyright (c) 2019 Athena. All rights reserved.
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

#include <athena/backend/llvm/runtime/structs.h>
#include <athena/core/inner/InnerFunctions.h>
#include <athena/ops/AddOperation.h>
#include <athena/core/Traversal.h>
#include <athena/core/FatalError.h>

#include <cassert>

using namespace athena::backend;
using namespace athena::core;
using namespace athena::core::inner;

namespace athena::ops {

void AddOperation::gen(AbstractGenerator &g, std::vector<Tensor *> &operationArguments) const {
    Tensor *c = operationArguments[2];
    Tensor *b = operationArguments[1];
    Tensor *a = operationArguments[0];
    g.generate("add", *a, *b, *c);
}

std::shared_ptr<Tensor> AddOperation::createTensor(
    core::Context& context, std::vector<Tensor *> args) const {
#ifdef DEBUG
    athena_assert(args.size() == 2, "AddOperation takes 2 arguments!");
    athena_assert(args[0]->getShape() == args[1]->getShape(), "Shapes of input tensors should be equals!");
    for (size_t argumentIndex = 0; argumentIndex < 2; ++argumentIndex) {
        athena_assert(args[argumentIndex]->getShape().getTotalSize() > 0, "Shape of tensor of argument ", argumentIndex, " should be exist!");
        athena_assert(args[argumentIndex]->getDataType() != DataType::UNDEFINED, "DataType of tensor of argument" , argumentIndex, " should be defined!");
        athena_assert(args[argumentIndex]->getSize() > 0, "Size of tensor of argument ", argumentIndex, " should be positive!");
        athena_assert(args[argumentIndex]->getVirtualAddress() > 0, "Virtual address of tensor of argument ", argumentIndex, " should be positive!");
    }
#endif
    return std::make_shared<Tensor>(args[0]->getDataType(), args[0]->getShape(), context);
}

void AddOperation::genIncomingDerivative(AbstractGenerator &g, std::vector<Tensor *> &operationArguments,
    Tensor &derivativeTensorOfIncomingNode, Tensor &derivativeTensorOfCurrentNode, size_t derivativeMark) const {
    float f_unit = 1.f;
    void *unit = reinterpret_cast<void *>(&f_unit);
#ifdef DEBUG
    // We need to make sure the derivative tensor exists
    assert(derivativeTensorOfIncomingNode.getDataType() != core::DataType::UNDEFINED &&
           "derivativeTensor is broken");
    assert(operationArguments.size() != 2 &&
        "arguments count is not 2");
#endif
    g.generate("fill", derivativeTensorOfIncomingNode, unit);
    static auto *options = new HadamardOptions<float>{1.f, 0.0f};
    void *opts = static_cast<void *>(options);
    g.generate("hadamard", opts, derivativeTensorOfIncomingNode, derivativeTensorOfCurrentNode,
               derivativeTensorOfIncomingNode);
}

}  // namespace athena::ops
