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

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
// fixme replace with MLIR-provided
inline void buildAffineLoopNest(
    OpBuilder& builder, Location loc, ValueRange lbs, ValueRange ubs,
    ArrayRef<int64_t> steps,
    function_ref<void(OpBuilder&, Location, ValueRange)> bodyBuilderFn) {
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Value, 4> ivs;
  ivs.reserve(lbs.size());

  for (unsigned i = 0, e = lbs.size(); i < e; ++i) {
    auto loopBody = [&](OpBuilder& nestedBuilder, Location nestedLoc,
                        Value iv) {
      ivs.push_back(iv);
      if (i == e - 1 && bodyBuilderFn) {
        OpBuilder::InsertionGuard nestedGuard(nestedBuilder);
        bodyBuilderFn(nestedBuilder, nestedLoc, ivs);
      }
      nestedBuilder.create<AffineTerminatorOp>(nestedLoc);
    };

    auto loop = builder.create<AffineForOp>(
        loc, lbs[i], builder.getDimIdentityMap(), ubs[i],
        builder.getDimIdentityMap(), steps[i]);
    builder.setInsertionPointToStart(loop.getBody());
    loopBody(builder, loc, loop.getInductionVar());
  }
}
} // namespace mlir
