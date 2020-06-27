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
void MatMulOp::produceKernel(OpBuilder& builder) {
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

  size_t kDim;
  if (false) {
    // if (transpose_left()) {
    kDim = 0;
  } else {
    kDim = 1;
  }

  mlir::Value K = builder.create<DimOp>(builder.getUnknownLoc(),
                                        kernel.getArgument(0), kDim);

  auto zero = builder.create<ConstantIntOp>(builder.getUnknownLoc(), 0,
                                            builder.getIndexType());
  auto one = builder.create<ConstantIntOp>(builder.getUnknownLoc(), 1,
                                           builder.getIndexType());

  builder.create<scf::ForOp>(
      builder.getUnknownLoc(), zero, K, one,
      ValueRange{kernel.getArgument(0), kernel.getArgument(1),
                 kernel.getArgument(2)},
      [this](OpBuilder& bld, Location loc, Value idx, ValueRange args) {
        mlir::Value leftRow, leftCol, rightRow, rightCol;

        if (false) {
          // if (transpose_left()) {
          leftRow = idx;
          leftCol = bld.create<compute::GlobalIdOp>(
              bld.getUnknownLoc(), bld.getIndexType(), bld.getIndexAttr(0));
        } else {
          leftCol = idx;
          leftRow = bld.create<compute::GlobalIdOp>(
              bld.getUnknownLoc(), bld.getIndexType(), bld.getIndexAttr(0));
        }

        if (false) {
          // if (transpose_right()) {
          rightCol = idx;
          rightRow = bld.create<compute::GlobalIdOp>(
              bld.getUnknownLoc(), bld.getIndexType(), bld.getIndexAttr(1));

        } else {
          rightRow = idx;
          rightCol = bld.create<compute::GlobalIdOp>(
              bld.getUnknownLoc(), bld.getIndexType(), bld.getIndexAttr(1));
        }

        mlir::Value leftVal = bld.create<LoadOp>(bld.getUnknownLoc(), args[0],
                                                 ValueRange{leftRow, leftCol});
        mlir::Value rightVal = bld.create<LoadOp>(
            bld.getUnknownLoc(), args[1], ValueRange{rightRow, rightCol});

        mlir::Value dim0 = bld.create<compute::GlobalIdOp>(bld.getUnknownLoc(),
                                                           bld.getIndexType(),
                                                           bld.getIndexAttr(0))
                               .getResult();
        mlir::Value dim1 = bld.create<compute::GlobalIdOp>(bld.getUnknownLoc(),
                                                           bld.getIndexType(),
                                                           bld.getIndexAttr(1))
                               .getResult();

        mlir::Value outVal = bld.create<LoadOp>(bld.getUnknownLoc(), args[2],
                                                ValueRange{dim0, dim1});

        mlir::Value mul =
            bld.create<MulFOp>(bld.getUnknownLoc(), leftVal, rightVal);
        mlir::Value sum = bld.create<AddFOp>(bld.getUnknownLoc(), mul, outVal);

        bld.create<StoreOp>(bld.getUnknownLoc(), sum, args[2],
                            ValueRange{dim0, dim1});
      });

  builder.create<compute::ReturnOp>(builder.getUnknownLoc());
}
} // namespace mlir::ath_graph
