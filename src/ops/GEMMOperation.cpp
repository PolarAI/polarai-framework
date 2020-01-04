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
#include <athena/ops/GEMMOperation.h>
#include <cassert>
#include <iostream>

using namespace athena::backend;
using namespace athena::core;
using namespace athena::core::inner;

namespace athena::ops {
std::shared_ptr<Tensor> GEMMOperation::createTensor(
    Context& context, std::vector<Tensor *> args) const {
    TensorShape shape{args[0]->getShape().dim(0), args[0]->getShape().dim(1)};
    return std::make_shared<Tensor>(args[0]->getDataType(), shape, context);
}

void GEMMOperation::gen(
    AbstractGenerator &g,
    std::vector<Tensor *> &operationArguments) const {
    void *opts;
    if (operationArguments[0]->getDataType() == DataType::FLOAT) {
        auto *options =
            new GEMMOptions<float>{mTransposeA, mTransposeB, 1.0f, 0.f};
        opts = static_cast<void *>(options);
    } else if (operationArguments[0]->getDataType() == DataType::DOUBLE) {
        auto *options =
            new GEMMOptions<double>{mTransposeA, mTransposeB, 1.0, 0.};
        opts = static_cast<void *>(options);
    } else {
        new FatalError(ATH_NOT_IMPLEMENTED, "Unsupported type");
    }
    g.generate("gemm", opts, *operationArguments[0], *operationArguments[1],
               *operationArguments[2]);
}

template <typename FPType>
void* GEMMOperation::createOptions(size_t derivativeMark) const {
    auto *options = new GEMMOptions<float>{false, false, 1.0f, 0.f};
    options->transposeB = derivativeMark == 0;
    options->transposeA = derivativeMark == 1;
    if (mTransposeB) options->transposeB = !options->transposeB;
    if (mTransposeA) options->transposeA = !options->transposeA;
    return static_cast<void *>(options);
}

void GEMMOperation::genIncomingDerivative(
    AbstractGenerator &g,
    std::vector<Tensor *> &operationArguments,
    Tensor &derivativeTensorOfIncomingNode,
    Tensor &derivativeTensorOfCurrentNode,
    size_t derivativeMark) const {
    void *opts;
    if (operationArguments[0]->getDataType() == DataType::FLOAT) {
        opts = createOptions<float>(derivativeMark);
    } else if (operationArguments[0]->getDataType() == DataType::DOUBLE) {
        opts = createOptions<double>(derivativeMark);
    } else {
        new FatalError(core::ATH_NOT_IMPLEMENTED, "Unsupported type");
    }

    inner::Tensor *tensorA, *tensorB;

    if (derivativeMark == 0) {
        tensorA = &derivativeTensorOfCurrentNode;
        tensorB = operationArguments[1];
    } else {
        tensorA = operationArguments[0];
        tensorB = &derivativeTensorOfCurrentNode;
    }
    g.generate("gemm", opts, *tensorA, *tensorB, derivativeTensorOfIncomingNode);
}

std::string GEMMOperation::serialize() const {
    std::stringstream stringstream;
    stringstream << mTransposeA << std::endl << mTransposeB << std::endl;
    return stringstream.str();
}
}  // namespace athena::ops