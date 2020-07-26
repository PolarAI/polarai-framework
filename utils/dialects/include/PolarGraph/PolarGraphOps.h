//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#ifndef POLAR_POLARGRAPHOPS_H
#define POLAR_POLARGRAPHOPS_H

#include "PolarGraph/ComputationalOpInterface.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "PolarGraph/PolarGraphOpsEnums.h.inc"

namespace mlir::polar_graph {
#define GET_OP_CLASSES
#include "PolarGraph/PolarGraphOps.h.inc"
} // namespace mlir::polar_graph

#endif // POLAR_POLARGRAPHOPS_H
