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

#ifndef ATHENA_IRCLEANUPPASS_H
#define ATHENA_IRCLEANUPPASS_H

#include <mlir/Pass/Pass.h>

namespace chaos {
class IRCleanUpPass
    : public mlir::OperationPass<IRCleanUpPass, mlir::ModuleOp> {
  void runOnOperation() override;
};

std::unique_ptr<IRCleanUpPass> createIRCleanUpPass();
} // namespace chaos

#endif // ATHENA_IRCLEANUPPASS_H
