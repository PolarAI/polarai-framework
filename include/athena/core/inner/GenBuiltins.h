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

#include "athena/core/Allocator.h"
#include <athena/core/inner/GenValues.h>

#include <functional>
#include <string_view>
#include <vector>

namespace athena::core {
enum class builtin {
  Alloc,        ///< Allocates memory for tensor.
  Lock,         ///< Locks tensor in memory.
  Release,      ///< Releases tensor memory.
  Barrier,      ///< Explicitly waits for all operations to complete.
  NodeEval,     ///< Evaluates node of a Graph.
  InvokeLoader, ///< Invokes loader routine.

  ///@{
  /// \name Operation builtins
  Add,      ///< Element-wise addition.
  Mul,      ///< Element-wise multiplication.
  MatMul,   ///< Matrix-matrix multiplication.
  Fill,     ///< Fill tensor with constant pattern.
  Slice,    ///< Get subtensor.
  Transpose /// Transpose 2D tensor (matrix).
  ///}
};

//===----------------------------------------------------------------------===//
// Builtin traits
//===----------------------------------------------------------------------===//

namespace inner {

template <builtin B> struct builtin_functor {};

template <builtin B>
using builtin_functor_t = typename builtin_functor<B>::type;

template <> struct builtin_functor<builtin::Alloc> {
  using type = std::function<GenValue(GenValue)>;
};

template <> struct builtin_functor<builtin::Lock> {
  using type = std::function<GenValue(GenValue, LockType)>;
};

template <> struct builtin_functor<builtin::Release> {
  using type = std::function<GenValue(GenValue)>;
};

template <> struct builtin_functor<builtin::InvokeLoader> {
  using type = std::function<GenValue(std::string_view, GenValue)>;
};

template <> struct builtin_functor<builtin::NodeEval> {
  using type =
      std::function<GenValue(GenGraph, GenNode, const std::vector<GenValue>&)>;
};

template <> struct builtin_functor<builtin::Add> {
  using type =
      std::function<GenValue(GenValue, GenValue, GenValue, GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::Mul> {
  using type =
      std::function<GenValue(GenValue, GenValue, GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::MatMul> {
  using type =
      std::function<GenValue(GenValue, GenValue, GenValue, GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::Fill> {
  using type =
      std::function<GenValue(GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::Slice> {
  using type =
      std::function<GenValue(GenValue, GenValue)>;
};

template <> struct builtin_functor<builtin::Transpose> {
  using type =
      std::function<GenValue(GenValue, GenValue)>;
};
} // namespace inner
} // namespace athena::core