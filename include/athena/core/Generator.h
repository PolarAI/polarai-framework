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

#ifndef ATHENA_GENERATOR_H
#define ATHENA_GENERATOR_H

#include <athena/core/AbstractLoader.h>
#include <athena/core/Context.h>
#include <athena/core/inner/Tensor.h>

#include <any>
#include <cstddef>
#include <functional>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

namespace athena::core {

struct GenValue {
  std::any value;
};
struct GenNode {
  std::any node;

  explicit GenNode(std::any node) : node(std::move(node)) {}
  GenNode(const GenNode& src) = default;
  GenNode(GenNode&& src) = default;
  GenNode& operator=(const GenNode&) = default;
  GenNode& operator=(GenNode&&) = default;
  virtual ~GenNode() = default;

  virtual auto getOperand(size_t) -> GenValue { return GenValue{}; }
  virtual auto getResult() -> GenValue { return GenValue{}; }
  virtual auto getBatchIndex() -> GenValue { return GenValue{}; }
};
struct GenInsertionPoint {
  std::any point;
};
struct GenGraph {
  std::any graph;
};

namespace builtin {
// Utility builtins.
constexpr static auto Alloc = "alloc";     ///< Allocates memory for tensor.
constexpr static auto Lock = "lock";       ///< Locks tensor in memory.
constexpr static auto Release = "release"; ///< Releases tensor memory.
constexpr static auto Barrier =
    "barrier"; ///< Explicitly waits for all operations to complete.
constexpr static auto NodeEval = "eval"; ///< Evaluates node of a graph.
constexpr static auto InvokeLoader =
    "invoke_loader"; ///< Invokes loader routine.

// Operation builtins.
constexpr static auto Add = "add";       ///< Element-wise addition.
constexpr static auto Mul = "mul";       ///< Element-wise multiplication.
constexpr static auto MatMul = "matmul"; ///< Matrix-matrix multiplication.
constexpr static auto Fill = "fill";     ///< Fill tensor with constant.
constexpr static auto Slice = "slice";   ///< Get subtensor.
constexpr static auto Transpose = "transpose"; ///< Transpose 2D tensor.
} // namespace builtin

// fixme move to utils directory.
template <typename Ret> struct AnyCallable {
  AnyCallable() {}
  template <typename... Args>
  AnyCallable(std::function<Ret(Args...)> fun) : mAny(fun) {}
  template <typename... Args> Ret operator()(Args&&... args) {
    return std::invoke(std::any_cast<std::function<Ret(Args...)>>(mAny),
                       std::forward<Args>(args)...);
  }
  std::any mAny;
};

template <> struct AnyCallable<void> {
  AnyCallable() {}
  template <typename... Args>
  AnyCallable(std::function<void(Args...)> fun) : mAny(fun) {}
  template <typename... Args> void operator()(Args&&... args) {
    std::invoke(std::any_cast<std::function<void(Args...)>>(mAny),
                std::forward<Args>(args)...);
  }
  std::any mAny;
};

/// A bridge between \c GraphCompiler and a backend.
class ATH_CORE_EXPORT Generator {
public:
  using SupportedConstantT =
      std::variant<int32_t, int64_t, uint32_t, uint64_t, float, double>;

  Generator() = default;

  /// Notifies generator to insert builtins at a particular point.
  void setInsertionPoint(const GenInsertionPoint& insertionPoint) {
    mSetInsertionPointFunc(insertionPoint);
  }

  void setInsertionPoint(const GenGraph& graph) {
    mSetGraphInsertionPointFunc(graph);
  }

  void setInsertionPoint(const GenNode& node) {
    mSetNodeInsertionPointFunc(node);
  }

  auto getInsertionPoint() const -> GenInsertionPoint {
    return mGetInsertionPointFunc();
  }

  /// Generates call to one of the predefined builtins.
  ///
  /// \tparam Args
  /// \param opcode is a name of builtin to generate call to.
  /// \param args are arguments, specific to the builtin.
  /// \return a backend-specific handle to builtin call result.
  template <typename... Args>
  auto callBuiltin(const std::string& opcode, Args&&... args) -> GenValue {
    if (mGeneratorFunctors.count(opcode) == 0) {
      new FatalError(ATH_FATAL_OTHER, "Call to undefined functor ", opcode);
    }
    auto functor = mGeneratorFunctors[opcode];
    return functor(args...);
  }

  /// Creates a node stub in IR.
  ///
  /// This can actually be noop for some backends.
  /// This member function does not update the insertion point.
  /// Calls from graph to node are not automatically generated.
  ///
  /// \param nodeName is a name of Node. Will be used for function name.
  /// \return a backend-specific handle to node.
  auto createNode(std::string_view nodeName, size_t nodeId, size_t clusterId,
                  std::vector<inner::Tensor>& args, inner::Tensor& outValue)
      -> GenNode {
    return mCreateNodeFunc(nodeName, nodeId, clusterId, args, outValue);
  }

  /// Creates a graph stub in IR.
  ///
  /// This can actually be noop for some backends.
  /// This member function does not update the insertion point.
  ///
  /// \param graphName is the name of graph to be generated. It may
  ///        be used for symbol name in IR.
  /// \param graphId is the ID of graph that is generated.
  auto createGraph(std::string_view graphName, size_t graphId) -> GenGraph {
    return mCreateGraphFunc(graphName, graphId);
  }

  auto createConstant(SupportedConstantT constant) -> GenValue {
    return mCreateConstantFunc(constant);
  }

  /// Registers a functor that generates a specific builtin.
  ///
  /// \param opcode is a name of builtin being generated.
  /// \param functor is a function object that generates specified builtin.
  template <typename... Args>
  void registerFunctor(const std::string& opcode,
                       std::function<GenValue(Args...)> functor) {
    if (mGeneratorFunctors.count(opcode)) {
      new FatalError(ATH_FATAL_OTHER, "Attempt to re-register functor ",
                     opcode);
    }

    mGeneratorFunctors.insert(
        {opcode, AnyCallable<GenValue>(std::move(functor))});
  }

  void
  registerConstantFunctor(std::function<GenValue(SupportedConstantT)> functor) {
    mCreateConstantFunc = std::move(functor);
  }

  void registerNodeFunctor(
      std::function<GenNode(std::string_view, size_t, size_t,
                            const std::vector<inner::Tensor>&, inner::Tensor&)>
          functor) {
    mCreateNodeFunc = std::move(functor);
  }

  void registerGraphFunctor(
      std::function<GenGraph(std::string_view, size_t)> functor) {
    mCreateGraphFunc = std::move(functor);
  }

  void registerSetInsertionPointFunctor(
      std::function<void(GenInsertionPoint)> functor) {
    mSetInsertionPointFunc = std::move(functor);
  }

  void registerSetInsertionPointFunctor(std::function<void(GenNode)> functor) {
    mSetNodeInsertionPointFunc = std::move(functor);
  }

  void registerSetInsertionPointFunctor(std::function<void(GenGraph)> functor) {
    mSetGraphInsertionPointFunc = std::move(functor);
  }

  void
  registerGetInsertionPointFunctor(std::function<GenInsertionPoint()> functor) {
    mGetInsertionPointFunc = std::move(functor);
  }

private:
  std::function<void(GenInsertionPoint)> mSetInsertionPointFunc;
  std::function<void(GenNode)> mSetNodeInsertionPointFunc;
  std::function<void(GenGraph)> mSetGraphInsertionPointFunc;
  std::function<GenInsertionPoint()> mGetInsertionPointFunc;
  std::unordered_map<std::string, AnyCallable<GenValue>> mGeneratorFunctors;
  std::function<GenNode(std::string_view, size_t, size_t,
                        const std::vector<inner::Tensor>&, inner::Tensor&)>
      mCreateNodeFunc;
  std::function<GenGraph(std::string_view, size_t)> mCreateGraphFunc;
  // todo support half constants
  std::function<GenValue(SupportedConstantT)> mCreateConstantFunc;
};
} // namespace athena::core

#endif // ATHENA_GENERATOR_H
