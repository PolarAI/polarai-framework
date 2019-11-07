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

#include <athena/core/inner/InnerFunctions.h>
#include <athena/ops/MSELossFunction.h>

#include <cassert>

namespace athena::ops {
core::inner::Tensor *ops::MSELossFunction::getResultTensor(
    core::Context& context, std::vector<core::inner::Tensor *> args) const {
    core::TensorShape newShape{1};
    return new core::inner::Tensor(args[0]->getDataType(), newShape, context);
}
core::inner::Tensor *MSELossFunction::getDerivativeTensor(
    core::Context& context, std::vector<core::inner::Tensor *> args, int argNo) const {
    return new core::inner::Tensor(args[0]->getDataType(), args[0]->getShape(), context);
}
void MSELossFunction::gen(
    core::AbstractGenerator &g,
    std::vector<core::inner::Tensor *> &operationArguments) const {
    g.generate("mse", *operationArguments[0], *operationArguments[1],
               *operationArguments[2]);
}
void MSELossFunction::genDerivative(
    int order,
    core::AbstractGenerator &g,
    core::inner::Tensor &operationResult,
    core::inner::Tensor &internalError,
    std::vector<core::inner::Tensor *> &operationArguments,
    core::inner::Tensor &derivativeTensor,
    int argNo) const {
#ifdef DEBUG
    assert(operationArguments.size() == 2 && "Operation args != 2");
#endif

    double scaleDouble = 2.0 / operationResult.getShapeView().getTotalSize();
    float scaleFloat = 2.0f / operationResult.getShapeView().getTotalSize();

    uint64_t scale = 0;
    uint64_t negScale = 0;

    switch (operationResult.getDataType()) {
        case core::DataType::DOUBLE: {
            double negScaleDouble = -scaleDouble;
            scale = *reinterpret_cast<uint64_t *>(&scaleDouble);
            negScale = *reinterpret_cast<uint64_t *>(&negScaleDouble);
            break;
        }
        case core::DataType::FLOAT: {
            float negScaleFloat = -scaleFloat;
            scale = *reinterpret_cast<uint64_t *>(&scaleFloat);
            negScale = *reinterpret_cast<uint64_t *>(&negScaleFloat);
            break;
        }
        default:
            new core::FatalError(core::ATH_NOT_IMPLEMENTED,
                                 "Data type not supported");
    }

    g.generate("fma", *operationArguments[0], scale, *operationArguments[1],
               negScale, derivativeTensor);
}
core::inner::Tensor *MSELossFunction::getErrorTensor(core::Context& context,
                                                     std::vector<core::inner::Tensor *> args, int derivativeOrder) const {
    // Loss node is always the last node (except for output)
    // No need for actual implementation
    // todo refactor me
    return nullptr;
}
std::string MSELossFunction::serialize() const {
    return "";
}
}  // namespace athena::ops
