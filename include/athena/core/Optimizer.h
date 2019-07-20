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

#include "AbstractGenerator.h"
namespace athena::core {
class Optimizer {
    public:
    virtual void genFix(AbstractGenerator &generator,
                        inner::Tensor &target,
                        inner::Tensor &error) = 0;
};
}  // namespace athena::core

#endif  // ATHENA_OPTIMIZER_H
