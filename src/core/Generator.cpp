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

#include <athena/core/Generator.h>

namespace athena::core {
void Generator::registerFunctor(const std::string& name, FunctorType functor) {
  mRegisteredFunctors[name] = std::move(functor);
}
void Generator::unregisterFunctor(const std::string& name) {
  if (hasFunctor(name)) {
    mRegisteredFunctors.erase(name);
  }
}
void Generator::generate(const std::string& functorName,
                         const std::vector<inner::Tensor>& args,
                         const std::any& options) {
  if (!hasFunctor(functorName))
    new FatalError(FatalErrorType::ATH_NOT_IMPLEMENTED, "No functor with name ",
                   functorName);

  mRegisteredFunctors[functorName](mContext, mGraphName, mNodeId, mClusterId,
                                   args, options);
}
void Generator::generateLoad(inner::Tensor& target, AbstractLoader* loader) {
  mRegisteredLoadGenerator(mContext, mGraphName, mNodeId, mClusterId, target,
                           loader);
}
} // namespace athena::core