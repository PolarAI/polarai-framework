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

#ifndef ATHENA_OPTIMIZER_H
#define ATHENA_OPTIMIZER_H

#include <athena/core/AbstractGenerator.h>
#include <athena/core/core_export.h>

namespace athena::core {
class ATH_CORE_EXPORT Optimizer {
    public:
    Optimizer() = default;
    Optimizer(const Optimizer &) = default;
    Optimizer(Optimizer &&) = default;
    virtual ~Optimizer() = default;
    [[nodiscard]] virtual size_t getRequiredOrder() const {
        return 0;
    };
    virtual void genFix(AbstractGenerator &generator,
                        inner::Tensor &target,
                        std::vector<inner::Tensor *> &errors){};
    virtual void genError(AbstractGenerator &generator,
                          std::vector<inner::Tensor *> &incomingDerivatives,
                          inner::Tensor &totalError){};
};
}  // namespace athena::core

#endif  // ATHENA_OPTIMIZER_H
