//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include "AthenaGraph/AthenaGraphOps.h"
#include "Compute/ComputeOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

namespace mlir::ath_graph {
void SigmoidOp::produceKernel(OpBuilder& builder) {
  auto eltType = out()
                     .getType()
                     .cast<RankedTensorType>()
                     .getElementType()
                     .cast<FloatType>();
  // TODO figure out the correct memory space
  llvm::SmallVector<Type, 3> inputs{MemRefType::get({-1}, eltType),
                                    MemRefType::get({-1}, eltType)};
  auto funcType = builder.getFunctionType(inputs, {});
  auto kernel = builder.create<compute::FuncOp>(builder.getUnknownLoc(),
                                                getKernelName(), funcType);
  OpBuilder::InsertionGuard guard{builder};
  builder.setInsertionPointToStart(&kernel.getBody().front());
  auto idx = builder.create<compute::GlobalIdOp>(
      builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));

  auto in = builder.create<LoadOp>(builder.getUnknownLoc(),
                                   kernel.getArgument(0), idx);
  auto negIn = builder.create<NegFOp>(builder.getUnknownLoc(), in);
  auto exp = builder.create<ExpOp>(builder.getUnknownLoc(), negIn);
  auto one = builder.create<ConstantFloatOp>(builder.getUnknownLoc(),
                                             APFloat(1.0f), eltType);
  auto sum = builder.create<AddFOp>(builder.getUnknownLoc(), one, exp);

  auto res = builder.create<DivFOp>(builder.getUnknownLoc(), one, sum);

  builder.create<StoreOp>(builder.getUnknownLoc(), res, kernel.getArgument(1),
                          idx.getResult());

  builder.create<compute::ReturnOp>(builder.getUnknownLoc());
}
} // namespace mlir::ath_graph

