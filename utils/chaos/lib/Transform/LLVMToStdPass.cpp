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

#include "LLVMToStdPass.h"
#include "ChaosDialect.h"
#include "LLVMToStandardConversionPattern.h"

#include <iostream>

using namespace mlir;

namespace chaos {

std::unique_ptr<LLVMToStdPass> createLLVMToStdPass(mlir::MLIRContext* ctx) {
  return std::make_unique<LLVMToStdPass>(ctx);
}

void LLVMToStdPass::runOnModule() {
  ModuleOp m = getModule();

  OwningRewritePatternList patterns;
  registerStdConversionPatterns(mTypeConverter, patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalOp<FuncOp>();
  target.addLegalOp<ConstantOp>();
  target.addLegalOp<ReinterpretOp>();

  if (failed(applyPartialConversion(m, target, patterns, &mTypeConverter)))
    signalPassFailure();
}
} // namespace chaos
