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

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

namespace mlir::polar_graph {
void MatMulOp::produceKernel(OpBuilder& builder, Block::BlockArgListType args) {
  auto memrefTy = args.back().getType().cast<MemRefType>();

  Value left, right;
  if (transpose_left()) {
    left = builder.create<linalg::TransposeOp>(
        builder.getUnknownLoc(), args[0],
        AffineMapAttr::get(builder.getMultiDimIdentityMap(memrefTy.getRank())));
  } else {
    left = args[0];
  }
  if (transpose_right()) {
    right = builder.create<linalg::TransposeOp>(
        builder.getUnknownLoc(), args[1],
        AffineMapAttr::get(builder.getMultiDimIdentityMap(memrefTy.getRank())));
  } else {
    right = args[1];
  }
  builder.create<linalg::MatmulOp>(builder.getUnknownLoc(), TypeRange{memrefTy},
                                   ValueRange{left, right, args[2]});
}
} // namespace mlir::polar_graph
