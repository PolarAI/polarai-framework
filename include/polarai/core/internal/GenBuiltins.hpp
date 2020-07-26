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

#pragma once

#include <polarai/core/internal/GenValues.hpp>
#include <polarai/core/loader/internal/TensorAllocator.hpp>

#include <functional>
#include <optional>
#include <string_view>
#include <tuple>
#include <vector>

namespace polarai::core::internal {
enum class builtin {
  Alloc,        ///< Allocates memory for tensor.
  Lock,         ///< Locks tensor in memory.
  Release,      ///< Releases tensor memory.
  Barrier,      ///< Explicitly waits for all operations to complete.
  NodeEval,     ///< Evaluates node of a Graph.
  InvokeLoader, ///< Invokes loader routine.
  Return,       ///< Returns a value from a node.

  ///@{
  /// \name Operation builtins
  Add,       ///< Element-wise addition.
  Conv2D,    ///< 2D convolution.
  Copy,      ///< Element-wise copying.
  Divide,    ///< Element-wise division.
  LogLoss,   ///< Element-wise logistic loss function.
  MatMul,    ///< Matrix-matrix multiplication.
  Mul,       ///< Element-wise multiplication.
  MulConcat, ///< Gradient concatenation.
  Pool2D,    ///< 2D pooling.
  Fill,      ///< Fill tensor with constant pattern.
  ReLU,      ///< Rectified linear unit.
  Sigmoid,   ///< Element-wise sigmoid.
  Softmax,   ///< Softmax.
  Slice,     ///< Get subtensor.
  Transpose  /// Transpose 2D tensor (matrix).
  ///}
};

//===----------------------------------------------------------------------===//
// Builtin traits
//===----------------------------------------------------------------------===//

template <builtin B> struct builtin_functor {
  // fixme change to void when we have a barrier.
  using type = int;
};

template <> struct builtin_functor<builtin::Alloc> {
  using type = std::function<GenValue(GenValue)>;
};

template <> struct builtin_functor<builtin::Lock> {
  using type = std::function<GenValue(GenValue, LockType)>;
};

template <> struct builtin_functor<builtin::Release> {
  using type = std::function<GenValue(GenValue)>;
};

template <> struct builtin_functor<builtin::Barrier> {
  using type = std::function<void(uint64_t)>;
};

template <> struct builtin_functor<builtin::InvokeLoader> {
  using type = std::function<GenValue(GenValue)>;
};

template <> struct builtin_functor<builtin::NodeEval> {
  using type =
      std::function<GenValue(GenGraph, GenNode, const std::vector<GenValue>&)>;
};

template <> struct builtin_functor<builtin::Return> {
  using type = std::function<void(std::optional<GenValue>)>;
};

template <> struct builtin_functor<builtin::Add> {
  using type =
      std::function<void(GenValue, GenValue, GenValue, GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::Conv2D> {
  using type = std::function<void(GenValue, GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::Copy> {
  using type = std::function<void(GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::Divide> {
  using type = std::function<void(GenValue, GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::LogLoss> {
  using type = std::function<void(GenValue, GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::MatMul> {
  using type = std::function<void(GenValue, GenValue, GenValue, bool, bool)>;
};

template <> struct builtin_functor<builtin::Mul> {
  using type = std::function<void(GenValue, GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::MulConcat> {
  using type = std::function<void(GenValue, GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::Pool2D> {
  using type =
      std::function<void(GenValue, GenValue, const std::vector<int64_t>&,
                         const std::vector<int64_t>&)>;
};

template <> struct builtin_functor<builtin::ReLU> {
  using type = std::function<void(GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::Sigmoid> {
  using type = std::function<void(GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::Softmax> {
  using type = std::function<void(GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::Fill> {
  using type = std::function<void(GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::Slice> {
  using type = std::function<void(GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::Transpose> {
  using type = std::function<void(GenValue, GenValue)>;
};

template <builtin B>
using builtin_functor_t = typename builtin_functor<B>::type;

// NOTE: It is important to preserve the same order as in builtin enum class.
using BuiltinMap = std::tuple<
    // clang-format off
    builtin_functor_t<builtin::Alloc>,
    builtin_functor_t<builtin::Lock>,
    builtin_functor_t<builtin::Release>,
    builtin_functor_t<builtin::Barrier>,
    builtin_functor_t<builtin::NodeEval>,
    builtin_functor_t<builtin::InvokeLoader>,
    builtin_functor_t<builtin::Return>,
    builtin_functor_t<builtin::Add>,
    builtin_functor_t<builtin::Conv2D>,
    builtin_functor_t<builtin::Copy>,
    builtin_functor_t<builtin::Divide>,
    builtin_functor_t<builtin::LogLoss>,
    builtin_functor_t<builtin::MatMul>,
    builtin_functor_t<builtin::Mul>,
    builtin_functor_t<builtin::MulConcat>,
    builtin_functor_t<builtin::Pool2D>,
    builtin_functor_t<builtin::Fill>,
    builtin_functor_t<builtin::ReLU>,
    builtin_functor_t<builtin::Sigmoid>,
    builtin_functor_t<builtin::Softmax>,
    builtin_functor_t<builtin::Slice>,
    builtin_functor_t<builtin::Transpose>>;
// clang-format on
} // namespace polarai::core::internal
