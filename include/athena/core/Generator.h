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

/// A bridge between \c GraphCompiler and a backend.
class ATH_CORE_EXPORT Generator {
public:
  using FunctorType = std::function<void(
      Context&,                          // Athena Context
      std::string_view,                  // Graph name
      size_t,                            // Id of node to generate for
      size_t,                            // Id of cluster where node belongs to
      const std::vector<inner::Tensor>&, // Arguments to the builtin
      const std::any&                    // Builtin-specific options
      )>;
  using LoadGenType =
      std::function<void(Context&,         // Athena Context
                         std::string_view, // Graph name
                         size_t,           // Id of node to generate for
                         size_t,         // Id of cluster where node belongs to
                         inner::Tensor&, // Target tensor
                         AbstractLoader* // Loader
                         )>;

private:
  std::unordered_map<std::string, FunctorType> mRegisteredFunctors;
  LoadGenType mRegisteredLoadGenerator;
  Context& mContext;
  std::string mGraphName{}; // Name of the graph to generate code for
  size_t mNodeId{};         // Id of the node to generate code for
  size_t mClusterId{};      // Id of the cluster where node belongs to

public:
  /// Constructs a Generator.
  ///
  /// \param ctx is an Athena context.
  /// \param state is used by functors to emit IR/code/etc. Its real type
  /// is defined by the backend.
  Generator(Context& ctx, std::any state) : mContext(ctx){};

  /// Registers functor for specific name.
  ///
  /// \param name is a name to associate functor to.
  /// \param functor is a functor type object (lambda, functor type, function
  /// pointer) that provides routines to generate code for builtin.
  void registerFunctor(const std::string& name, FunctorType functor);

  /// @return true if a functor with name is registered.
  [[nodiscard]] bool hasFunctor(const std::string& name) const {
    return mRegisteredFunctors.count(name);
  }

  /// Removes any functor associated with the specified name.
  ///
  /// \param name is a functor name.
  void unregisterFunctor(const std::string& name);

  void setLoadGenerator(LoadGenType generator) {
    mRegisteredLoadGenerator = std::move(generator);
  }

  /// Sets coordinates for code generation.
  ///
  /// \param graphName is a name of a Graph, that all the following calls
  ///        are generated for.
  /// \param nodeId is an ID of a node, that all the following calls are
  ///        generated for.
  /// \param clusterId is an ID of a cluster, that node belongs to.
  void setGenerationPoint(const std::string& graphName, size_t nodeId,
                          size_t clusterId) {
    mGraphName = graphName;
    mNodeId = nodeId;
    mClusterId = clusterId;
  }

  /// Calls functor corresponding to provided builtin name.
  ///
  /// \param functorName is a name of builtin to generate code for.
  /// \param args is a vector of tensors that contains both arguments and
  ///        resulting tensor.
  /// \param options is an optional builtin options object.
  void generate(const std::string& functorName,
                const std::vector<inner::Tensor>& args,
                const std::any& options = nullptr);

  /// Calls routine to generate an invokation of loader.
  ///
  /// \param target is a Tensor, that stores load results.
  /// \param loader is a loader, that performs actual job.
  void generateLoad(inner::Tensor& target, AbstractLoader* loader);
};
} // namespace athena::core

#endif // ATHENA_GENERATOR_H
