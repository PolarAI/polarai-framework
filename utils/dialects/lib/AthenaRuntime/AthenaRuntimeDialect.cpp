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

#include "AthenaRuntime/AthenaRuntimeDialect.h"
#include "AthenaRuntime/AthenaRuntimeOps.h"

#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::ath_rt;

AthenaRuntimeDialect::AthenaRuntimeDialect(mlir::MLIRContext* context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<DeviceType, EventType, GraphHandleType>();
  addOperations<
#define GET_OP_LIST
#include "AthenaRuntime/AthenaRuntimeOps.cpp.inc"
      >();
}

mlir::Type AthenaRuntimeDialect::parseType(mlir::DialectAsmParser& parser) const {
  if (!parser.parseOptionalKeyword("device")) {
    return DeviceType::get(getContext());
  } else if (!parser.parseOptionalKeyword("event")) {
    return EventType::get(getContext());
  } else if (!parser.parseOptionalKeyword("graph_handle")) {
    return GraphHandleType::get(getContext());
  } else {
    return mlir::Type{};
  }
}

void AthenaRuntimeDialect::printType(mlir::Type type,
                                     mlir::DialectAsmPrinter& printer) const {
  if (type.isa<DeviceType>()) {
    printer << "device";
  } else if (type.isa<EventType>()) {
    printer << "event";
  } else if (type.isa<GraphHandleType>()) {
    printer << "graph_handle";
  }
}
