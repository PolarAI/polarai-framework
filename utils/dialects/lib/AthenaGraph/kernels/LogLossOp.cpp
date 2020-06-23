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
void LogLossOp::produceKernel(OpBuilder& builder) {
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
  auto idx = builder.create<compute::GlobalIdOp>(
      builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));
  auto groundTruth = builder.create<LoadOp>(
      builder.getUnknownLoc(), kernel.getArgument(0), idx.getResult());
  auto predicted = builder.create<LoadOp>(
      builder.getUnknownLoc(), kernel.getArgument(1), idx.getResult());
  auto negGT = builder.create<NegFOp>(builder.getUnknownLoc(), groundTruth);
  auto eps = builder.create<ConstantFloatOp>(builder.getUnknownLoc(),
                                             APFloat(1e-5), eltType);
  auto sum1 = builder.create<AddFOp>(builder.getUnknownLoc(), predicted, eps);
  auto log1 = builder.create<LogOp>(builder.getUnknownLoc(), sum1);
  auto mul1 = builder.create<MulFOp>(builder.getUnknownLoc(), negGT, log1);
  auto one = builder.create<ConstantFloatOp>(builder.getUnknownLoc(),
                                             llvm::APFloat(1.), eltType);
  auto sub1 = builder.create<SubFOp>(builder.getUnknownLoc(), one, predicted);
  auto sum2 = builder.create<AddFOp>(builder.getUnknownLoc(), sub1, eps);
  auto log2 = builder.create<LogOp>(builder.getUnknownLoc(), sum2);
  auto sub2 = builder.create<SubFOp>(builder.getUnknownLoc(), one, groundTruth);
  auto mul2 = builder.create<MulFOp>(builder.getUnknownLoc(), sub2, log2);
  auto res = builder.create<SubFOp>(builder.getUnknownLoc(), mul1, mul2);

  builder.create<StoreOp>(builder.getUnknownLoc(), res, kernel.getArgument(2),
                          idx.getResult());

  builder.create<compute::ReturnOp>(builder.getUnknownLoc());
}
} // namespace mlir::ath_graph
