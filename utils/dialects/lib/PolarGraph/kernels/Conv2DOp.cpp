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
#include "mlir/IR/Builders.h"

namespace mlir::polar_graph {
// Fixme only single channel is supported
void Conv2DOp::produceKernel(OpBuilder& builder, Block::BlockArgListType args) {
  auto memrefTy = args.back().getType().cast<MemRefType>();
  auto tensorTy = out().getType().cast<RankedTensorType>();
  auto kernelTensorTy = conv_kernel().getType().cast<RankedTensorType>();
  auto zero = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 0);

  SmallVector<Value, 3> lbs(memrefTy.getRank(), zero);
  SmallVector<Value, 3> ubs;
  SmallVector<int64_t, 3> steps(memrefTy.getRank(), 1);

  for (int i = 0; i < memrefTy.getRank(); i++) {
    auto dim = builder.create<ConstantIndexOp>(builder.getUnknownLoc(),
                                               tensorTy.getDimSize(i));
    ubs.push_back(dim);
  }
  int64_t offset0 = kernelTensorTy.getDimSize(0) / 2;
  int64_t offset1 = kernelTensorTy.getDimSize(1) / 2;
  auto bodyBuilder = [args, &memrefTy, offset0, offset1](
                         OpBuilder& builder, Location loc, ValueRange idx) {
    auto fzero = builder.create<ConstantFloatOp>(
        loc, APFloat(0.f), memrefTy.getElementType().cast<FloatType>());
    builder.create<AffineStoreOp>(loc, fzero, args[2], idx);
    SmallVector<int64_t, 2> lbs = {-offset0, -offset1};
    SmallVector<int64_t, 2> ubs = {offset0 + 1, offset1 + 1};
    SmallVector<int64_t, 2> steps(2, 1);
    auto innerBuilder = [args, offset0, offset1, outerIdx = idx](
                            OpBuilder& builder, Location loc, ValueRange idx) {
      auto constOffset0 = builder.create<ConstantIndexOp>(loc, offset0);
      auto constOffset1 = builder.create<ConstantIndexOp>(loc, offset0);
      auto idx0 = builder.create<SubFOp>(loc, idx[0], constOffset0);
      auto idx1 = builder.create<SubFOp>(loc, idx[1], constOffset1);
      auto weight =
          builder.create<AffineLoadOp>(loc, args[1], ValueRange{idx0, idx1});

      auto outIdx0 = builder.create<SubFOp>(loc, outerIdx[0], constOffset0);
      auto outIdx1 = builder.create<SubFOp>(loc, outerIdx[1], constOffset1);

      auto inp = builder.create<AffineLoadOp>(loc, args[0],
                                              ValueRange{outIdx0, outIdx1});
      auto out =
          builder.create<AffineLoadOp>(loc, args[2], ValueRange{outerIdx});

      auto mul = builder.create<MulFOp>(loc, inp, weight);
      auto sum = builder.create<AddFOp>(loc, mul, out);

      builder.create<AffineStoreOp>(loc, sum, args[2], outerIdx);
    };
    buildAffineLoopNest(builder, builder.getUnknownLoc(), lbs, ubs, steps,
                        innerBuilder);
  };
  buildAffineLoopNest(builder, builder.getUnknownLoc(), lbs, ubs, steps,
                      bodyBuilder);
}
} // namespace mlir::polar_graph
