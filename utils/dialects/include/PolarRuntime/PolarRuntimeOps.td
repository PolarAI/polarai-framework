#ifndef POLAR_RUNTIME_OPS
#define POLAR_RUNTIME_OPS

include "PolarRuntimeDialect.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LaunchFuncOp : PolarRuntime_Op<"launch_func"> {
  let summary = "Launches kernel on a specific device";
  let description = [{ TBD }];

  let arguments = (ins PolarRuntime_DeviceType:$device, 
                       PolarRuntime_EventType:$blocking_event,
                       StrAttr:$kernel, I64ArrayAttr:$global_size,
                       I64ArrayAttr:$local_size,
                       Variadic<AnyType>:$operands);
  let results = (outs PolarRuntime_EventType:$out_event);
}

def DeviceSelectOp : PolarRuntime_Op<"select_device", [NoSideEffect]> {
  let summary = "Selects device for specified node";
  let description = [{ TBD }];

  let arguments = (ins IndexAttr:$nodeId);
  let results = (outs PolarRuntime_DeviceType);
}

def AllocOp : PolarRuntime_Op<"alloc"> {
  let summary = "Allocates tensor on a particular device";
  
  let arguments = (ins PolarRuntime_DeviceType:$device, AnyTensor:$tensor);
}

def LockOp : PolarRuntime_Op<"lock"> {
  let summary = "Locks tensor memory on a particular device";

  let arguments = (ins PolarRuntime_DeviceType:$device, AnyTensor:$tensor, 
                   StrAttr:$lock_type);
}

def ReleaseOp : PolarRuntime_Op<"release"> {
  let summary = "Indicates that tensor memory can be freed";

  let arguments = (ins PolarRuntime_DeviceType:$device, AnyTensor:$tensor,
                   Variadic<PolarRuntime_EventType>:$event);
}

def BarrierOp : PolarRuntime_Op<"barrier"> {
  let summary = "Wait for all the events to finish";
  let description = [{ TBD }];

  let arguments = (ins Variadic<PolarRuntime_EventType>:$events);
}

def ScopeOp : PolarRuntime_Op<"scope", [
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

def ScopeYieldOp : PolarRuntime_Op<"yield", [NoSideEffect, HasParent<"ScopeOp">,
                                             Terminator]> {
  let summary = "Default scope terminator";
  let description = [{ TBD }];
}

def NullEventOp : PolarRuntime_Op<"null_event", [NoSideEffect]> {
  let summary = "Returns null event";

  let results = (outs PolarRuntime_EventType);
}
#endif // POLAR_RUNTIME_OPS
