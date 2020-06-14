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

#ifndef ATHENA_COMPUTETOOPENCLSPIRVPASS_H
#define ATHENA_COMPUTETOOPENCLSPIRVPASS_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

#include <memory>

namespace mlir {
// Forward declaration
class SPIRVTypeConverter;
class ModuleOp;
template <typename T> class OperationPass;

void populateComputeToOpenCLSPIRVPatterns(MLIRContext* context,
                                          SPIRVTypeConverter& typeConverter,
                                          OwningRewritePatternList& patterns);

auto createConvertComputeToOpenCLSPIRVPass()
    -> std::unique_ptr<OperationPass<ModuleOp>>;
} // namespace mlir

#endif // ATHENA_COMPUTETOOPENCLSPIRVPASS_H
