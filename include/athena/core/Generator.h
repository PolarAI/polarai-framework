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
#include <functional>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace athena::core {

using GenValue = std::any;
using GenNode = std::any;
using GenInsertionPoint = std::any;
using GenGraph = std::any;
using GenFunction = std::any;

namespace builtin {
constexpr auto Alloc = "alloc";
constexpr auto Lock = "lock";
constexpr auto Release = "release";
constexpr auto Barrier = "barrier";
constexpr auto NodeEval = "eval";
constexpr auto Call = "call";
constexpr auto Load = "load";
constexpr auto Store = "store";
constexpr auto InvokeLoader = "invoke_loader";

constexpr auto Add = "add";
constexpr auto Mul = "mul";
constexpr auto MatMul = "matmul";
constexpr auto Fill = "fill";
constexpr auto Slice = "slice";
constexpr auto Transpose = "transpose";
} // namespace builtin

/// A bridge between \c GraphCompiler and a backend.
class ATH_CORE_EXPORT Generator {
public:
  Generator() = default;

  template <typename... Args>
  void
  registerFunctor(const std::string& opcode,
                  std::function<GenValue(GenInsertionPoint, Args...)> functor) {
    if (mGeneratorFunctors.count(opcode)) {
      new FatalError(ATH_FATAL_OTHER, "Attempt to re-register functor ",
                     opcode);
    }

    mGeneratorFunctors.insert({opcode, functor});
  }

  template <typename FuncLikeT>
  void setInsertionPointToBegin(FuncLikeT funcLike);

  template <typename FuncLikeT>
  void setInsertionPointToEnd(FuncLikeT funcLike);

  void setInsertionPoint(GenInsertionPoint insertionPoint) {
    mInsertionPoint = std::move(insertionPoint);
  }

  /// Generates call to one of the predefined builtins.
  ///
  /// \tparam Args
  /// \param opcode is a name of builtin to generate call to.
  /// \param args are arguments, specific to the builtin.
  /// \return a backend-specific handle to builtin call result.
  template <typename... Args>
  GenValue callBuiltin(const std::string& opcode, Args&&... args) {
    if (mGeneratorFunctors.count(opcode) == 0) {
      new FatalError(ATH_FATAL_OTHER, "Call to undefined functor ", opcode);
    }
    auto functor =
        std::any_cast<std::function<GenValue(GenInsertionPoint, Args...)>>(
            mGeneratorFunctors[opcode]);
    return functor(mInsertionPoint, std::forward<Args>(args)...);
  }

  /// Creates a node stub in IR.
  ///
  /// This can actually be noop for some backends.
  ///
  /// \param nodeName is a name of Node. Will be used for function name.
  /// \return a backend-specific handle to node.
  GenNode createNode(std::string_view nodeName, size_t nodeId, size_t clusterId,
                     std::vector<inner::Tensor>& args,
                     inner::Tensor& outValue) {
    return mCreateNodeFunc(nodeName, nodeId, clusterId, args, outValue);
  }

  GenGraph createGraph(std::string_view graphName, size_t graphId) {
    return mCreateGraphFunc(graphName, graphId);
  }

  template <typename ...Types>
  GenFunction createFreeFunction(std::string_view functionName) {
    // todo capture function types.
  }

  template <typename Callee>
  GenValue createCall(Callee callee, std::vector<GenValue>& args) {
    // todo there must be some way to differentiate two callees.
  }

  template <typename T>
  GenValue createVariable(std::string_view varName) {
    // todo catch type
  }

  template <typename T>
  GenValue createConstant(std::string_view constName) {
    // todo catch type
  }

  std::tuple<GenInsertionPoint, GenValue> createLoop(GenValue lowerBound, GenValue upperBound) {
    return mCreateLoopFunc(lowerBound, upperBound);
  }

private:
  GenInsertionPoint mInsertionPoint;
  std::unordered_map<std::string, std::any> mGeneratorFunctors;
  std::function<GenNode(std::string_view, size_t, size_t,
                        std::vector<inner::Tensor>&, inner::Tensor&)> mCreateNodeFunc;
  std::function<GenGraph(std::string_view, size_t)> mCreateGraphFunc;
  std::function<GenValue(GenFunction, std::vector<GenValue>&)> mCreateFunctionCallFunc;
  std::function<GenValue(GenGraph, std::vector<GenValue>&)> mCreateGraphCallFunc;
  std::function<std::tuple<GenInsertionPoint, GenValue>(GenValue, GenValue)> mCreateLoopFunc;
};
} // namespace athena::core

#endif // ATHENA_GENERATOR_H
