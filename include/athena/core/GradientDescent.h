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

#ifndef ATHENA_GRADIENTDESCENT_H
#define ATHENA_GRADIENTDESCENT_H

#include <athena/core/Optimizer.h>
#include <athena/core/core_export.h>

namespace athena::core {
class ATH_CORE_EXPORT GradientDescent : public Optimizer {
protected:
  double mLearningRate;

public:
  GradientDescent() : mLearningRate(0.01){};
  explicit GradientDescent(double learningRate)
      : Optimizer(), mLearningRate(learningRate){};
  ~GradientDescent() override = default;
  [[nodiscard]] size_t getRequiredOrder() const override;
  void genFix(AbstractGenerator& generator, inner::Tensor& target,
              std::vector<inner::Tensor*>& errors) override;
  void genError(AbstractGenerator& generator,
                std::vector<inner::Tensor*>& incomingDerivatives,
                inner::Tensor& totalError) override;
};
} // namespace athena::core

#endif // ATHENA_GRADIENTDESCENT_H
