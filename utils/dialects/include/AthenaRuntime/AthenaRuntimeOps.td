#ifndef ATHENA_RUNTIME_OPS
#define ATHENA_RUNTIME_OPS

include "AthenaRuntimeDialect.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffects.td"

def AthenaRuntime_LaunchOp : AthenaRuntime_Op<"launch"> {
  let summary = "Launches kernel on a specific device";
  let description = [{ TBD }];

  let arguments = (ins AthenaRuntime_DeviceType:$device,
                       StrAttr:$kernel, Variadic<AnyType>:$operands);
  let results = (outs AthenaRuntime_EventType);
}

def AthenaRuntime_DeviceSelectOp : AthenaRuntime_Op<"select_device", 
                                                               [NoSideEffect]> {
  let summary = "Selects device for specified node";
  let description = [{ TBD }];

  let arguments = (ins IndexAttr:$nodeId);
  let results = (outs AthenaRuntime_DeviceType);
}

def AthenaRuntime_BarrierOp : AthenaRuntime_Op<"barrier"> {
  let summary = "Wait for all the events to finish";
  let description = [{ TBD }];

  let arguments = (ins Variadic<AthenaRuntime_EventType>:$events);
}

def AthenaRuntime_ScopeOp : AthenaRuntime_Op<"scope", [
                               SingleBlockImplicitTerminator<"ScopeYieldOp">]> {
  let summary = "Defines a parallel scope";
  let description = [{ TBD }];

  let arguments = (ins Index:$batchSize);

  let regions = (region SizedRegion<1>:$region);
  
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder& builder, OperationState& result, Value size">
  ];
}

def AthenaRuntime_ScopeYieldOp : AthenaRuntime_Op<"yield", [NoSideEffect, 
                                                        HasParent<"ScopeOp">,
                                                        Terminator]> {
  let summary = "Default scope terminator";
  let description = [{ TBD }];
}

def AthenaRuntime_NullEventOp : AthenaRuntime_Op<"null_event", [NoSideEffect]> {
  let summary = "Returns null event";

  let results = (outs AthenaRuntime_EventType);
}
#endif // ATHENA_RUNTIME_OPS
