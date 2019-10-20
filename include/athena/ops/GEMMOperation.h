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

#ifndef ATHENA_GEMMOPERATION_H
#define ATHENA_GEMMOPERATION_H

#include <athena/core/Operation.h>
namespace athena::ops {
class GEMMOperation : public core::Operation {
    private:
    bool mTransposeA;
    bool mTransposeB;

    public:
    GEMMOperation(bool transposeA, bool transposeB)
        : Operation("gemm"),
          mTransposeA(transposeA), mTransposeB(transposeB) {}
    core::inner::Tensor *getResultTensor(
        std::vector<core::inner::Tensor *> args) const override;
    core::inner::Tensor *getErrorTensor(std::vector<core::inner::Tensor *> args,
                                        int derivativeOrder) const override;
    core::inner::Tensor *getDerivativeTensor(
        std::vector<core::inner::Tensor *> args, int argNo) const override;
    void gen(
        core::AbstractGenerator &g,
        std::vector<core::inner::Tensor *> &operationArguments) const override;
    void genDerivative(int order,
                       core::AbstractGenerator &g,
                       core::inner::Tensor &operationResult,
                       core::inner::Tensor &internalError,
                       std::vector<core::inner::Tensor *> &operationArguments,
                       core::inner::Tensor &derivativeTensor,
                       int argNo) const override;
    size_t getOperandsCount() const override {
        return 2;
    }
};
}  // namespace athena::ops

#endif  // ATHENA_GEMMOPERATION_H
