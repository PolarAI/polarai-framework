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

#include <cassert>

using namespace athena::backend;

namespace athena::ops {

void AddOperation::gen(
    core::AbstractGenerator &g,
    std::vector<core::inner::Tensor *> &operationArguments) const {
    core::inner::Tensor *c = operationArguments[2];
    core::inner::Tensor *b = operationArguments[1];
    core::inner::Tensor *a = operationArguments[0];

    g.generate("add", *a, *b, *c);
}

core::inner::Tensor *AddOperation::createTensor(
    core::Context& context, std::vector<core::inner::Tensor *> args, int argNo) const {
#ifdef DEBUG
    assert(argNo < 2 && "AddOperation takes 2 arguments!");
#endif
    return new core::inner::Tensor(args[argNo]->getDataType(),
                                   args[argNo]->getShape(), context);
}
void AddOperation::genIncomingDerivatives(
    const int order,
    core::AbstractGenerator &g,
    core::inner::Tensor &operationResult,
    core::inner::Tensor &internalError,
    std::vector<core::inner::Tensor *> &operationArguments,
    core::inner::Tensor &derivativeTensor,
    int argNo) const {
    float f_unit = 1;
    void *unit = reinterpret_cast<void *>(&f_unit);
#ifdef DEBUG
    // We need to make sure the derivative tensor exists
    assert(derivativeTensor.getDataType() != core::DataType::UNDEFINED &&
           "derivativeTensor is broken");
#endif
    // todo this is a workaround because I'm too lazy to implement proper copy
    static auto *options = new HadamardOptions<float>{1.f, 0.0f};
    void *opts = static_cast<void *>(options);
    g.generate("fill", derivativeTensor, unit);
    g.generate("hadamard", opts, derivativeTensor, internalError,
               derivativeTensor);
}

}  // namespace athena::ops
