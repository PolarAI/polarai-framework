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
#include <athena/ops/ops_export.h>

namespace athena::ops {
class ATH_OPS_EXPORT GEMMOperation : public core::Operation {
    private:
    bool mTransposeA;
    bool mTransposeB;

    public:
    GEMMOperation(bool transposeA, bool transposeB)
        : Operation("gemm"),
          mTransposeA(transposeA), mTransposeB(transposeB) {}
    std::shared_ptr<core::inner::Tensor> createTensor(
        core::Context& context, std::vector<core::inner::Tensor *> args) const override;
    void gen(
        core::AbstractGenerator &g,
        std::vector<core::inner::Tensor *> &operationArguments) const override;
    void genIncomingDerivative(
        core::AbstractGenerator &g,
        std::vector<core::inner::Tensor *> &operationArguments,
        core::inner::Tensor &derivativeOfIncomingNode,
        core::inner::Tensor &ownDerivative,
        size_t argumentMark) const override;
    size_t getOperandsCount() const override {
        return 2;
    }
    std::string serialize() const override;

    static Operation *deserialize(const std::string &data) {
        std::stringstream stream(data);
        bool transpA, transpB;
        stream >> transpA;
        stream >> transpB;
        return new GEMMOperation(transpA, transpB);
    };

    private:
    template <typename FPType>
    void* createOptions(size_t derivativeMark) const;
};
}  // namespace athena::ops

#endif  // ATHENA_GEMMOPERATION_H
