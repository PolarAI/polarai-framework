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

#include "PolarGraph/PolarGraphOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

namespace mlir::polar_graph {
void MatMulOp::produceKernel(OpBuilder& builder, Block::BlockArgListType args) {
  auto memrefTy = args.back().getType().cast<MemRefType>();
  auto zero = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 0);

  SmallVector<Value, 3> lbs(memrefTy.getRank(), zero);
  SmallVector<Value, 3> ubs;
  SmallVector<int64_t, 3> steps(memrefTy.getRank(), 1);

  for (int i = 0; i < memrefTy.getRank(); i++) {
    auto dim = builder.create<ConstantIndexOp>(builder.getUnknownLoc(),
                                               memrefTy.getDimSize(i));
    ubs.push_back(dim);
  }

  auto bodyBuilder = [args, this, memrefTy](OpBuilder& builder, Location loc,
                                            ValueRange idx) {
    size_t kDim;
    if (transpose_left()) {
      kDim = 0;
    } else {
      kDim = 1;
    }
    auto innerLoop =
        builder.create<AffineForOp>(loc, 0, memrefTy.getDimSize(kDim), 1);

    builder.setInsertionPointToStart(innerLoop.getBody());

    mlir::Value kIdx = innerLoop.getInductionVar();
    mlir::Value leftRow, leftCol, rightRow, rightCol;

    if (transpose_left()) {
      leftRow = kIdx;
      leftCol = idx[0];
    } else {
      leftCol = kIdx;
      leftRow = idx[0];
    }

    if (transpose_right()) {
      rightCol = kIdx;
      rightRow = idx[1];
    } else {
      rightRow = kIdx;
      rightCol = idx[1];
    }

    mlir::Value leftVal = builder.create<AffineLoadOp>(
        loc, args[0], ValueRange{leftRow, leftCol});
    mlir::Value rightVal = builder.create<AffineLoadOp>(
        loc, args[1], ValueRange{rightRow, rightCol});

    mlir::Value outVal = builder.create<AffineLoadOp>(loc, args[2], idx);

    mlir::Value mul = builder.create<MulFOp>(loc, leftVal, rightVal);
    mlir::Value sum =
        builder.create<AddFOp>(builder.getUnknownLoc(), mul, outVal);

    builder.create<AffineStoreOp>(loc, sum, args[2], idx);
  };
  buildAffineLoopNest(builder, builder.getUnknownLoc(), lbs, ubs, steps,
                      bodyBuilder);
}
} // namespace mlir::polar_graph
