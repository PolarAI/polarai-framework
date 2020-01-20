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

#include "IRCleanUpPass.h"
#include <mlir/IR/Function.h>

using namespace mlir;

namespace chaos {
void IRCleanUpPass::runOnOperation() {
  ModuleOp module = getOperation();

  for (auto& func : module) {
    auto attrs = func.getAttrs();
    for (auto& attr : attrs) {
      if (attr.first == "safe_to_remove") {
        auto boolAttr = attr.second.cast<BoolAttr>();
        if (boolAttr.getValue()) {
          func.erase();
        }
      }
    }
  }
}

std::unique_ptr<IRCleanUpPass> createIRCleanUpPass() {
  return std::make_unique<IRCleanUpPass>();
}
} // namespace chaos
