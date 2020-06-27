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

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

namespace mlir::ath_graph {
void MatMulConcatOp::produceKernel(OpBuilder& builder) {
  auto eltType = out()
                     .getType()
                     .cast<RankedTensorType>()
                     .getElementType()
                     .cast<FloatType>();
  // TODO figure out the correct memory space
  llvm::SmallVector<Type, 3> inputs{MemRefType::get({-1}, eltType),
                                    MemRefType::get({-1}, eltType),
                                    MemRefType::get({-1}, eltType)};
  auto funcType = builder.getFunctionType(inputs, {});
  auto kernel = builder.create<compute::FuncOp>(builder.getUnknownLoc(),
                                                getKernelName(), funcType);
  OpBuilder::InsertionGuard guard{builder};
  builder.setInsertionPointToStart(&kernel.getBody().front());
  auto m = builder.create<compute::GlobalIdOp>(
      builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));
  auto n = builder.create<compute::GlobalIdOp>(
      builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(1));

  mlir::Value K;

  auto zero = builder.create<ConstantFloatOp>(builder.getUnknownLoc(),
                                              APFloat(0.), eltType);
  auto grad = builder.create<LoadOp>(builder.getUnknownLoc(),
                                     kernel.getArgument(0), idx.getResult());
  auto localDeriv = builder.create<LoadOp>(
      builder.getUnknownLoc(), kernel.getArgument(1), idx.getResult());
  auto res = builder.create<MulFOp>(builder.getUnknownLoc(), grad, localDeriv);
  builder.create<StoreOp>(builder.getUnknownLoc(), res, kernel.getArgument(2),
                          idx.getResult());
  builder.create<compute::ReturnOp>(builder.getUnknownLoc());
}
} // namespace mlir::ath_graph
