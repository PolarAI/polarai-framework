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

#include <support/types.hpp>

// fixme support CblasConjTrans, AtlasConj
// fixme support CblasColMajor

namespace picomath {

/// A generic kernel to implement matrix-matrix multiplication.
///
/// This kernel aims to be a portable GEMM implementation. No optimizations
/// applied. Data structures are templates to allow kernel re-usage in
/// different programming models.
///
/// \tparam DataT is the data type of matrix elements.
template <typename DataT> class GemmKernel {
private:
  using InAccT = global_accessor<DataT, 2, access_mode::read>;
  using OutAccT = global_accessor<DataT, 2, access_mode::write>;
  InAccT mBufA;
  InAccT mBufB;
  OutAccT mResBuf;
  const size_t M, N, K;
  bool mTransposeA, mTransposeB;

public:
  GemmKernel(InAccT bufA, bool transposeA, InAccT bufB, bool transposeB,
             OutAccT resBuf, size_t M, size_t N, size_t K)
      : mBufA(bufA), mTransposeA(transposeA), mBufB(bufB),
        mTransposeB(transposeB), mResBuf(resBuf), M(M), N(N), K(K) {}
  void operator()(id<2> idx) {
    typename OutAccT::value_type acc = 0;

    for (size_t k = 0; k < K; k++) {
      id<2> aId, bId; // todo replace with constexpr

      if (mTransposeA) {
        aId = {k, idx[0]};
      } else {
        aId = {idx[0], k};
      }

      if (mTransposeB) {
        bId = {idx[1], k};
      } else {
        bId = {k, idx[1]};
      }

      acc += mBufA[aId] * mBufB[bId];
    }
    mResBuf[idx] = acc;
  }
};

template <typename DataT>
GemmKernel(global_accessor<const DataT, 2, access_mode::read>, bool,
           global_accessor<const DataT, 2, access_mode::read>, bool,
           global_accessor<DataT, 2, access_mode::write>, size_t, size_t,
           size_t) -> GemmKernel<DataT>;
} // namespace picomath
