//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

namespace mlir::polar_graph {
// Fixme only single channel is supported
void SoftmaxOp::produceKernel(OpBuilder& builder,
                              Block::BlockArgListType args) {
  auto memrefTy = args.back().getType().cast<MemRefType>();
  auto tensorTy = out().getType().cast<RankedTensorType>();
  auto zero = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 0);

  SmallVector<Value, 3> lbs(memrefTy.getRank(), zero);
  SmallVector<Value, 3> ubs;
  SmallVector<int64_t, 3> steps(memrefTy.getRank(), 1);

  for (int i = 0; i < memrefTy.getRank(); i++) {
    auto dim = builder.create<ConstantIndexOp>(builder.getUnknownLoc(),
                                               tensorTy.getDimSize(i));
    ubs.push_back(dim);
  }

  auto bodyBuilder = [args, &memrefTy, lbs, ubs,
                      steps](OpBuilder& builder, Location loc, ValueRange idx) {
    auto fzero = builder.create<ConstantFloatOp>(
        loc, APFloat(0.f), memrefTy.getElementType().cast<FloatType>());
    builder.create<AffineStoreOp>(loc, fzero, args[1], idx);

    auto reduceBody = [args, outerIdx = idx](OpBuilder& builder, Location loc,
                                             ValueRange idx) {
      auto inp = builder.create<AffineLoadOp>(loc, args[0], idx);
      auto exp = builder.create<ExpOp>(loc, inp);

      auto out = builder.create<AffineLoadOp>(loc, args[1], outerIdx);

      auto res = builder.create<AddFOp>(loc, exp, out);

      builder.create<StoreOp>(loc, res, args[1], outerIdx);
    };
    buildAffineLoopNest(builder, builder.getUnknownLoc(), lbs, ubs, steps,
                        reduceBody);
    auto total = builder.create<AffineLoadOp>(loc, args[1], idx);
    auto inp = builder.create<AffineLoadOp>(loc, args[0], idx);

    auto exp = builder.create<ExpOp>(loc, inp);

    auto div = builder.create<DivFOp>(loc, exp, total);

    builder.create<AffineStoreOp>(loc, div, args[1], idx);
  };
  buildAffineLoopNest(builder, builder.getUnknownLoc(), lbs, ubs, steps,
                      bodyBuilder);
}
} // namespace mlir::polar_graph
