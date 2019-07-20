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

#include <athena/core/inner/InnerFunctions.h>
#include <athena/ops/AddOperation.h>

#include <cassert>

namespace athena::ops {

void AddOperation::gen(
    core::AbstractGenerator &g,
    std::vector<core::inner::Tensor *> &operationArguments) const {
    core::inner::Tensor *c = operationArguments[2];
    core::inner::Tensor *b = operationArguments[1];
    core::inner::Tensor *a = operationArguments[0];

    g.generate("add", *a, *b, *c);
}
core::inner::Tensor &AddOperation::getResultTensor(
    std::vector<core::inner::Tensor *> args) const {
    core::ShapeView shapeView(args[0]->getShapeView());
    return core::inner::createTensor(args[0]->getDataType(),
                                     shapeView.toShape());
}

core::inner::Tensor &AddOperation::getDerivativeTensor(
    std::vector<core::inner::Tensor *> args, int argNo) const {
#ifdef DEBUG
    assert(argNo < 2 && "AddOperation takes 2 arguments!");
#endif
    core::ShapeView shapeView(args[argNo]->getShapeView());
    return core::inner::createTensor(args[argNo]->getDataType(),
                                     shapeView.toShape());
}
void AddOperation::genDerivative(
    core::AbstractGenerator &g,
    std::vector<core::inner::Tensor *> &operationArguments,
    int argNo) const {
    float f_unit = 1;
    void *unit = reinterpret_cast<void *>(&f_unit);
    // We need to make sure the 4th (3rd in terms of vector)
    // tensor persist and is a derivative tensor
#ifdef DEBUG
    assert(operationArguments.size() >= 3 &&
           "operationArguments[2] must be derivative tensor");
    assert(operationArguments[2]->getDataType() != core::DataType::UNDEFINED &&
           "operationArguments[2] is broken");
#endif
    g.generate("fill", *operationArguments[2], unit);
}

}  // namespace athena::ops
