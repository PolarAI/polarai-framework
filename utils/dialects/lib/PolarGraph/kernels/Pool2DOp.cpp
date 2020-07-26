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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"

namespace mlir::polar_graph {
// Fixme only single channel is supported
void Pool2DOp::produceKernel(OpBuilder& builder, Block::BlockArgListType args) {
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

  SmallVector<int64_t, 2> windowSize;
  for (auto& attr : window().getValue()) {
    windowSize.push_back(attr.cast<IntegerAttr>().getInt());
  }
  SmallVector<int64_t, 2> strides;
  for (auto& attr : stride().getValue()) {
    strides.push_back(attr.cast<IntegerAttr>().getInt());
  }
  auto bodyBuilder = [args, &memrefTy, windowSize, strides](
                         OpBuilder& builder, Location loc, ValueRange idx) {
    auto verySmallNumber = builder.create<ConstantFloatOp>(
        loc, APFloat(-10e5f), memrefTy.getElementType().cast<FloatType>());
    builder.create<AffineStoreOp>(loc, verySmallNumber, args[1], idx);
    SmallVector<int64_t, 2> lbs(2, 0);
    const SmallVector<int64_t, 2>& ubs = windowSize;
    SmallVector<int64_t, 2> steps(2, 1);
    auto innerBuilder = [args, windowSize, strides, outerIdx = idx](
                            OpBuilder& builder, Location loc, ValueRange idx) {
      auto idxDim = getAffineDimExpr(0, builder.getContext());
      auto outerIdxDim = getAffineDimExpr(1, builder.getContext());
      auto strideSym = getAffineSymbolExpr(0, builder.getContext());

      auto expr = outerIdxDim * strideSym + idxDim;

      auto map = AffineMap::get(2, 1, expr);

      auto stride0 = builder.create<ConstantIndexOp>(loc, strides[0]);
      auto stride1 = builder.create<ConstantIndexOp>(loc, strides[1]);

      auto idx0 = builder.create<AffineApplyOp>(
          loc, map, ValueRange{idx[0], outerIdx[0], stride0});
      auto idx1 = builder.create<AffineApplyOp>(
          loc, map, ValueRange{idx[1], outerIdx[1], stride1});

      auto inp =
          builder.create<AffineLoadOp>(loc, args[0], ValueRange{idx0, idx1});

      auto out = builder.create<AffineLoadOp>(loc, args[1], outerIdx);

      auto cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, inp, out);
      auto res = builder.create<SelectOp>(loc, cmp, inp, out);
      builder.create<AffineStoreOp>(loc, res, args[1], outerIdx);
    };
    buildAffineLoopNest(builder, builder.getUnknownLoc(), lbs, ubs, steps,
                        innerBuilder);
  };
  buildAffineLoopNest(builder, builder.getUnknownLoc(), lbs, ubs, steps,
                      bodyBuilder);
}
} // namespace mlir::polar_graph
