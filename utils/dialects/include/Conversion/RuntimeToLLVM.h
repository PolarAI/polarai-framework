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

#ifndef ATHENA_RUNTIMETOLLVM_H
#define ATHENA_RUNTIMETOLLVM_H

#include <memory>

namespace mlir {

class ModuleOp;
class MLIRContext;
class OwningRewritePatternList;

template <typename OpT> class OperationPass;

void populateRuntimeToLLVMConversionPatterns(
    
    OwningRewritePatternList& loweringPatterns, MLIRContext* ctx);

auto createLowerRuntimeToLLVMPass()
    -> std::unique_ptr<OperationPass<ModuleOp>>;

} // namespace mlir

#endif // ATHENA_RUNTIMETOLLVM_H

