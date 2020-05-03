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

#ifndef ATHENA_PASSES_H
#define ATHENA_PASSES_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename OpT> class OperationPass;

std::unique_ptr<OperationPass<ModuleOp>> createDeployDefaultFunctionsPass();
}

#endif // ATHENA_PASSES_H
