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

#ifndef ATHENA_ATHENARUNTIMEDIALECT_H
#define ATHENA_ATHENARUNTIMEDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser.h"

namespace mlir::ath_rt {

namespace RuntimeTypes {
enum Types {
  Device = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_9_TYPE,
  Event,
  GraphHandle
};
}

class DeviceType : public Type::TypeBase<DeviceType, Type> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == RuntimeTypes::Device; }

  static DeviceType get(MLIRContext* context) {
    return Base::get(context, RuntimeTypes::Device);
  }
};

class EventType : public Type::TypeBase<EventType, Type> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == RuntimeTypes::Event; }

  static EventType get(MLIRContext* context) {
    return Base::get(context, RuntimeTypes::Event);
  }
};

class GraphHandleType : public Type::TypeBase<GraphHandleType, Type> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == RuntimeTypes::GraphHandle; }

  static GraphHandleType get(MLIRContext* context) {
    return Base::get(context, RuntimeTypes::GraphHandle);
  }
};

#include "AthenaRuntime/AthenaRuntimeOpsDialect.h.inc"
} // namespace mlir::ath_graph

#endif // ATHENA_RUNTIMEDIALECT_H
