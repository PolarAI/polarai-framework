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

using namespace athena::core;
using namespace athena::backend;

namespace athena::ops {
inner::Tensor *GEMMOperation::getResultTensor(
    core::Context &context, std::vector<core::inner::Tensor *> args) const {
    size_t m = args[0]->getShapeView().dim(mTransposeA ? 1 : 0);
    size_t k = args[0]->getShapeView().dim(mTransposeA ? 0 : 1);
    size_t n = args[0]->getShapeView().dim(mTransposeB ? 0 : 1);

#ifdef DEBUG
    size_t k2 = args[0]->getShapeView().dim(mTransposeA ? 0 : 1);
    assert(k == k2 &&
           "Number of columns of A must be equal to number of rows of B");
#endif

    TensorShape shape{m, n};

    return new core::inner::Tensor(args[0]->getDataType(), shape, context);
}
core::inner::Tensor *GEMMOperation::getDerivativeTensor(
    core::Context &context,
    std::vector<core::inner::Tensor *> args,
    int argNo) const {
    core::ShapeView shapeView(args[argNo]->getShapeView());
    return new core::inner::Tensor(args[argNo]->getDataType(),
                                   shapeView.toShape(), context);
}
void GEMMOperation::gen(
    core::AbstractGenerator &g,
    std::vector<core::inner::Tensor *> &operationArguments) const {
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
void GEMMOperation::genDerivative(
    int order,
    core::AbstractGenerator &g,
    core::inner::Tensor &operationResult,
    core::inner::Tensor &internalError,
    std::vector<core::inner::Tensor *> &operationArguments,
    core::inner::Tensor &derivativeTensor,
    int argNo) const {
    void *opts;

#ifdef DEBUG
    assert(order == 1 && "Higher orders are not supported");
#endif

    if (operationArguments[0]->getDataType() == DataType::FLOAT) {
        auto *options = new GEMMOptions<float>{false, false, 1.0f, 0.f};
        options->transposeB = argNo == 0;
        options->transposeA = argNo == 1;
        if (mTransposeB) options->transposeB = !options->transposeB;
        if (mTransposeA) options->transposeA = !options->transposeA;
        opts = static_cast<void *>(options);
    } else if (operationArguments[0]->getDataType() == DataType::DOUBLE) {
        auto *options = new GEMMOptions<double>{false, false, 1.0, 0.};
        options->transposeA = argNo == 0;
        options->transposeB = argNo == 1;
        if (mTransposeB) options->transposeB = !options->transposeB;
        if (mTransposeA) options->transposeA = !options->transposeA;
        opts = static_cast<void *>(options);
    } else {
        new FatalError(core::ATH_NOT_IMPLEMENTED, "Unsupported type");
    }

    inner::Tensor *tensorA, *tensorB;

    if (argNo == 0) {
        tensorA = &internalError;
        tensorB = operationArguments[1];
    } else {
        tensorA = operationArguments[0];
        tensorB = &internalError;
    }

    g.generate("gemm", opts, *tensorA, *tensorB, derivativeTensor);
}
core::inner::Tensor *GEMMOperation::getErrorTensor(
    core::Context &context,
    std::vector<core::inner::Tensor *> args,
    int) const {
    // todo higher orders not supported
    return getResultTensor(context, args);
}
std::string GEMMOperation::serialize() const {
    std::stringstream stringstream;
    stringstream << mTransposeA << std::endl << mTransposeB << std::endl;
    return stringstream.str();
}
}