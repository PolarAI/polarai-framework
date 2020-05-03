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

#ifndef ATHENA_GRAPHTORUNTIMEPASS_H
#define ATHENA_GRAPHTORUNTIMEPASS_H

#include <memory>

namespace mlir {

class ModuleOp;
class MLIRContext;
class OwningRewritePatternList;

template <typename OpT> class OperationPass;

void populateGraphToRuntimeConversionPatterns(
    OwningRewritePatternList& structureLoweringPatterns,
    OwningRewritePatternList& nodeOpsLoweringPatterns, MLIRContext* ctx);

std::unique_ptr<OperationPass<ModuleOp>> createLowerGraphToRuntimePass();

} // namespace mlir

#endif // ATHENA_GRAPHTORUNTIMEPASS_H
