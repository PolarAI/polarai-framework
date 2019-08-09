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

namespace athena::ops {
core::inner::Tensor &ops::MSELossFunction::getResultTensor(
    std::vector<core::inner::Tensor *> args) const {
    core::TensorShape newShape{1};
    return core::inner::createTensor(args[0]->getDataType(), newShape);
}
core::inner::Tensor &MSELossFunction::getDerivativeTensor(
    std::vector<core::inner::Tensor *> args, int argNo) const {
    core::TensorShape newShape{1};
    return core::inner::createTensor(args[0]->getDataType(), newShape);
}
void MSELossFunction::gen(
    core::AbstractGenerator &g,
    std::vector<core::inner::Tensor *> &operationArguments) const {
    g.generate("mse", *operationArguments[0], *operationArguments[1]);
}
void MSELossFunction::genDerivative(
    int order,
    core::AbstractGenerator &g,
    core::inner::Tensor &operationResult,
    std::vector<core::inner::Tensor *> &operationArguments,
    core::inner::Tensor &derivativeTensor,
    int argNo) const {
    double scaleDouble = 2.0 / operationResult.getShapeView().getTotalSize();

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
            double negScaleDouble = -scaleDouble;
            scale = *reinterpret_cast<uint64_t *>(&scaleDouble);
            negScale = *reinterpret_cast<uint64_t *>(&negScaleDouble);
            break;
        }
        default:
            new core::FatalError(1, "Data type not supported");
    }

    g.generate("fma", *operationArguments[0], scale, &operationArguments[1],
               negScale, derivativeTensor);
}
}  // namespace athena::ops
