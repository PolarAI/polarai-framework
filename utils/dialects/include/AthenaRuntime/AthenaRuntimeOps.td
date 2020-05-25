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
                       AthenaRuntime_EventType:$blocking_event,
                       StrAttr:$kernel, I64ArrayAttr:$global_size,
                       I64ArrayAttr:$local_size,
                       Variadic<AnyType>:$operands);
  let results = (outs AnyTensor:$out_tensor,
                      AthenaRuntime_EventType:$out_event);
}

def AthenaRuntime_DeviceSelectOp : AthenaRuntime_Op<"select_device", 
                                                               [NoSideEffect]> {
  let summary = "Selects device for specified node";
  let description = [{ TBD }];

  let arguments = (ins IndexAttr:$nodeId);
  let results = (outs AthenaRuntime_DeviceType);
}

def AthenaRuntime_AllocOp : AthenaRuntime_Op<"alloc"> {
  let summary = "Allocates tensor on a particular device";
  
  let arguments = (ins AthenaRuntime_DeviceType:$device, AnyTensor:$tensor);
}

def AthenaRuntime_LockOp : AthenaRuntime_Op<"lock"> {
  let summary = "Locks tensor memory on a particular device";

  let arguments = (ins AthenaRuntime_DeviceType:$device, AnyTensor:$tensor, 
                   StrAttr:$lock_type);
}

def AthenaRuntime_ReleaseOp : AthenaRuntime_Op<"release"> {
  let summary = "Indicates that tensor memory can be freed";

  let arguments = (ins AthenaRuntime_DeviceType:$device, AnyTensor:$tensor);
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
