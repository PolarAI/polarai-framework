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

#include <athena/core/internal/GenBuiltins.h>
#include <athena/core/internal/GenValues.h>
#include <athena/core/tensor/internal/TensorInternal.h>

#include <any>
#include <cstddef>
#include <functional>
#include <memory>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#include <athena/core/core_export.h>

namespace athena::core::internal {
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
  template <builtin B, typename... Args>
  auto callBuiltin(Args&&... args) {
    internal::builtin_functor_t<B>& functor =
        std::get<static_cast<int>(B)>(mGeneratorFunctors);
    return functor(std::forward<Args>(args)...);
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
                  std::vector<internal::TensorInternal*>& args,
                  internal::TensorInternal& outValue) -> GenNode {
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
  /// \tparam B is a builtin being generated.
  /// \param functor is a function object that generates specified builtin.
  template <builtin B>
  void registerFunctor(internal::builtin_functor_t<B>& functor) {
    std::get<static_cast<int>(B)>(mGeneratorFunctors) = std::move(functor);
  }

  void
  registerConstantFunctor(std::function<GenValue(SupportedConstantT)> functor) {
    mCreateConstantFunc = std::move(functor);
  }

  void registerNodeFunctor(
      std::function<GenNode(std::string_view, size_t, size_t,
                            const std::vector<internal::TensorInternal*>&,
                            internal::TensorInternal&)>
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
  BuiltinMap mGeneratorFunctors;
  std::function<GenNode(std::string_view, size_t, size_t,
                        const std::vector<internal::TensorInternal*>&,
                        internal::TensorInternal&)>
      mCreateNodeFunc;
  std::function<GenGraph(std::string_view, size_t)> mCreateGraphFunc;
  // todo support half constants
  std::function<GenValue(SupportedConstantT)> mCreateConstantFunc;
};
} // namespace athena::core::internal

#endif // ATHENA_GENERATOR_H
