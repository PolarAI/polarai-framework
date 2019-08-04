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

#include <athena/core/AbstractGenerator.h>
#include <athena/core/GradientDescent.h>
#include <athena/core/inner/Tensor.h>

namespace athena::core {
size_t GradientDescent::getRequiredOrder() const {
    return 1;
}
void athena::core::GradientDescent::genFix(
    AbstractGenerator &generator,
    inner::Tensor &target,
    std::vector<inner::Tensor *> &errors) {}
void athena::core::GradientDescent::genErrors(
    AbstractGenerator &generator,
    std::vector<inner::Tensor *> &derivativeTensors,
    std::vector<inner::Tensor *> &nodeErrorTensors,
    std::vector<inner::Tensor *> &outcomingErrorTensors) {}
}  // namespace athena::core
