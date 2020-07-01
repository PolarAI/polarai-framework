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
namespace polar_rt {
class ApplyOp;
}
class ModuleOp;
class FuncOp;
template <typename OpT> class OperationPass;

std::unique_ptr<OperationPass<ModuleOp>> createDeployDefaultFunctionsPass();
std::unique_ptr<OperationPass<ModuleOp>> createGraphRelationDestructorPass();
std::unique_ptr<OperationPass<FuncOp>> createBarrierLegalizerPass();
std::unique_ptr<OperationPass<FuncOp>> createLegalizeRTForLoweringPass();
auto createReleaseDependencyPass() -> std::unique_ptr<OperationPass<FuncOp>>;
auto createRuntimeShapeInferencePass()
    -> std::unique_ptr<OperationPass<FuncOp>>;
auto createKernelMaterializerPass() -> std::unique_ptr<OperationPass<FuncOp>>;
auto createKernelOutliningPass() -> std::unique_ptr<OperationPass<ModuleOp>>;
} // namespace mlir

#endif // ATHENA_PASSES_H
