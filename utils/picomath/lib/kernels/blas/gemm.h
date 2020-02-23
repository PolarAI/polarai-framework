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

#include <iostream>

// fixme support CblasConjTrans, AtlasConj
// fixme support CblasColMajor

/// A generic kernel to implement matrix-matrix multiplication.
///
/// This kernel aims to be a portable GEMM implementation. No optimizations
/// applied. Data structures are templates to allow kernel re-usage in
/// different programming models.
///
/// \tparam AccessorAT is an accessor type for matrix A.
/// \tparam AccessorBT is an accessor type for matrix B.
/// \tparam AccessorCT is an accessor type for matrix C.
/// \tparam IdT is index type.
template <typename AccessorAT, typename AccessorBT, typename AccessorCT,
          typename IdT>
class GemmKernel {
private:
  AccessorAT mBufA;
  AccessorBT mBufB;
  AccessorCT mResBuf;
  const size_t M, N, K;
  bool mTransposeA, mTransposeB;

public:
  GemmKernel(AccessorAT bufA, bool transposeA, AccessorBT bufB, bool transposeB,
             AccessorCT resBuf, size_t M, size_t N, size_t K)
      : mBufA(bufA), mTransposeA(transposeA), mBufB(bufB),
        mTransposeB(transposeB), mResBuf(resBuf), M(M), N(N), K(K) {}
  void operator()(IdT id) {
    typename AccessorCT::value_type acc = 0;

    for (size_t k = 0; k < K; k++) {
      IdT aId, bId; // todo replace with constexpr

      if (mTransposeA) {
        aId = {k, id[0]};
      } else {
        aId = {id[0], k};
      }

      if (mTransposeB) {
        bId = {id[1], k};
      } else {
        bId = {k, id[1]};
      }

      acc += mBufA[aId] * mBufB[bId];
    }
    mResBuf[id] = acc;
  }
};
