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
#include <athena/backend/llvm/runtime/structs.h>

#include <cassert>

using namespace athena::backend;
using namespace athena::core;
using namespace athena::core::inner;

namespace athena::ops {
void MSELossFunction::gen(AbstractGenerator &g, std::vector<Tensor *> &operationArguments) const {
    g.generate("mse", *operationArguments[0], *operationArguments[1], *operationArguments[2]);
}

std::shared_ptr<Tensor> MSELossFunction::createTensor(Context& context, std::vector<Tensor *> args) const {
    return std::make_shared<Tensor>(Tensor(args[0]->getDataType(), {1}, context));
}

void MSELossFunction::genIncomingDerivative(AbstractGenerator &g, std::vector<core::inner::Tensor *> &operationArguments,
    Tensor &derivativeTensorOfIncomingNode, Tensor &derivativeTensorOfCurrentNode,
    size_t derivativeMark) const {
#ifdef DEBUG
    assert(operationArguments.size() == 2 && "Operation args != 2");
#endif

    uint64_t scale = 0;
    uint64_t negScale = 0;

    switch (operationArguments[0]->getDataType()) {
        case core::DataType::DOUBLE: {
            double scaleDouble = 2.0;
            double negScaleDouble = -scaleDouble;
            scale = *reinterpret_cast<uint64_t *>(&scaleDouble);
            negScale = *reinterpret_cast<uint64_t *>(&negScaleDouble);
            break;
        }
        case core::DataType::FLOAT: {
            float scaleFloat = 2.0f;
            float negScaleFloat = -scaleFloat;
            scale = *reinterpret_cast<uint64_t *>(&scaleFloat);
            negScale = *reinterpret_cast<uint64_t *>(&negScaleFloat);
            break;
        }
        default:
            new core::FatalError(core::ATH_NOT_IMPLEMENTED,
                                 "Data type not supported");
    }
    size_t otherArg = derivativeMark == 0 ? 1 : 0;
    g.generate("fma", *operationArguments[derivativeMark], scale, *operationArguments[otherArg],
               negScale, derivativeTensorOfIncomingNode);
    static auto *options = new backend::HadamardOptions<float>{1.f, 0.0f};
    void *opts = static_cast<void *>(options);
    g.generate("hadamard", opts, derivativeTensorOfIncomingNode, derivativeTensorOfCurrentNode,
               derivativeTensorOfIncomingNode);
}

std::string MSELossFunction::serialize() const {
    return "";
}
}  // namespace athena::ops
