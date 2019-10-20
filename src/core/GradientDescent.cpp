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

#include <athena/backend/llvm/runtime/structs.h>
#include <athena/core/AbstractGenerator.h>
#include <athena/core/GradientDescent.h>
#include <athena/core/inner/Tensor.h>

using namespace athena::backend;

namespace athena::core {
size_t GradientDescent::getRequiredOrder() const {
    return 1;
}
void athena::core::GradientDescent::genFix(
    AbstractGenerator &generator,
    inner::Tensor &target,
    std::vector<inner::Tensor *> &errors) {
    float fltUnit = 1.0;
    double dblUnit = 1.0;
    uint64_t unit = 0;
    uint64_t learningRate;

    switch (target.getDataType()) {
        case DataType::DOUBLE:
            unit = *reinterpret_cast<uint64_t *>(&dblUnit);
            learningRate = *reinterpret_cast<uint64_t *>(&mLearningRate);
            break;
        case DataType::FLOAT: {
            unit = *reinterpret_cast<uint64_t *>(&fltUnit);
            auto fltLR = static_cast<float>(mLearningRate);
            learningRate = *reinterpret_cast<uint64_t *>(&fltLR);
            break;
        }
        default:
            new FatalError(1, "Unsupported type");
    }

    for (auto *errTensor : errors) {
        generator.generate("fma", *errTensor, learningRate, target, unit,
                           target);
    }
}
void athena::core::GradientDescent::genError(
    AbstractGenerator &generator,
    std::vector<inner::Tensor *> &incomingDerivatives,
    inner::Tensor &totalError) {
    double dblZero = 0.0;
    float fltZero = 0.0;
    uint64_t zero;

    switch (totalError.getDataType()) {
        case DataType::DOUBLE: {
            zero = *reinterpret_cast<uint64_t *>(&dblZero);
            break;
        }
        case DataType::FLOAT: {
            zero = *reinterpret_cast<uint64_t *>(&fltZero);
            break;
        }
        default:
            new FatalError(-1, "Unsupported type");
    }

    void *pzero = static_cast<void *>(&zero);
    generator.generate("fill", totalError, pzero);

    for (auto &incomingDerivative : incomingDerivatives) {
        generator.generate("add", totalError, *incomingDerivative, totalError);
    }
}
}  // namespace athena::core
