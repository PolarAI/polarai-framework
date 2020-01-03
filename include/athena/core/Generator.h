/*
 * Copyright (c) 2020 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_GENERATOR_H
#define ATHENA_GENERATOR_H

#include <athena/core/Context.h>
#include <athena/core/inner/Tensor.h>

#include <any>
#include <functional>
#include <unordered_map>
#include <utility>

namespace athena::core {

/// A bridge between \c GraphCompiler and a backend.
class ATH_CORE_EXPORT Generator {
    public:
    using FunctorType = std::function<void(Context &,
                                           std::any &,
                                           size_t,
                                           const std::string &,
                                           size_t,
                                           const std::vector<inner::Tensor> &,
                                           const std::any &)>;

    private:
    std::unordered_map<std::string, FunctorType> mRegisteredFunctors;
    Context &mContext;
    std::any mGeneratorState;

    public:
    /// Constructs a Generator.
    ///
    /// \param ctx is an Athena context.
    /// \param state is used by functors to emit IR/code/etc. Its real type
    /// is defined by the backend.
    Generator(Context &ctx, std::any state)
        : mContext(ctx), mGeneratorState(std::move(state)){};

    /// Registers functor for specific name.
    ///
    /// \param name is a name to associate functor to.
    /// \param functor is a functor type object (lambda, functor type, function
    /// pointer) that provides routines to generate code for builtin.
    void registerFunctor(const std::string &name, FunctorType functor);

    /// @return true if a functor with name is registered.
    [[nodiscard]] bool hasFunctor(const std::string &name) const {
        return mRegisteredFunctors.count(name);
    }

    /// Removes any functor associated with the specified name.
    ///
    /// \param name is a functor name.
    void unregisterFunctor(const std::string &name);

    /// Calls functor corresponding to provided builtin name.
    ///
    /// \param functorName is a name of builtin to generate code for.
    /// \param nodeId is an identifier of a node that code is generated for.
    /// \param nodeName is a name of a node that code is generated for.
    /// \param clusterId is an identifier of a cluster that contains a node that
    /// code is generated for.
    /// \param args is a vector of tensors that contains both arguments and
    /// resulting tensor.
    /// \param options is an optional builtin options object.
    void generate(const std::string &functorName,
                  size_t nodeId,
                  const std::string &nodeName,
                  size_t clusterId,
                  const std::vector<inner::Tensor> &args,
                  const std::any &options = nullptr);
};
}  // namespace athena::core

#endif  // ATHENA_GENERATOR_H
