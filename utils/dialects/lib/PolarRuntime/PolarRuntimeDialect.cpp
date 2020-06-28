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

#include "PolarRuntime/PolarRuntimeDialect.h"
#include "PolarRuntime/PolarRuntimeOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::polar_rt;

PolarRuntimeDialect::PolarRuntimeDialect(mlir::MLIRContext* context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<DeviceType, EventType, GraphHandleType>();
  addOperations<
#define GET_OP_LIST
#include "PolarRuntime/PolarRuntimeOps.cpp.inc"
      >();
}

mlir::Type
PolarRuntimeDialect::parseType(mlir::DialectAsmParser& parser) const {
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

void PolarRuntimeDialect::printType(mlir::Type type,
                                    mlir::DialectAsmPrinter& printer) const {
  if (type.isa<DeviceType>()) {
    printer << "device";
  } else if (type.isa<EventType>()) {
    printer << "event";
  } else if (type.isa<GraphHandleType>()) {
    printer << "graph_handle";
  }
}
