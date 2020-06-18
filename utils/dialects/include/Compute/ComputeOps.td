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

include "Compute/ComputeDialect.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Compute_ModuleOp : Compute_Op<"module", [IsolatedFromAbove, SymbolTable, Symbol,
                                             SingleBlockImplicitTerminator<"ModuleEndOp">]> {
  let summary = "A top level compilation unit for offload code";
  let description = [{
    TBD
  }];

  let builders = [OpBuilder<"OpBuilder& builder, OperationState &result, "
                            "StringRef name">];

  let regions = (region SizedRegion<1>:$body);
  let skipDefaultBuilders = 1;
}

def Compute_ModuleEndOp : Compute_Op<"module_end", [Terminator, HasParent<"ModuleOp">]> {
  let summary = "A pseudo operation that marks end of a compute.module";
  let description = [{
    TBD
  }];

  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
}

def Compute_FuncOp : Compute_Op<"func", [HasParent<"ModuleOp">, AutomaticAllocationScope,
                                         FunctionLike, IsolatedFromAbove, Symbol]> {
  let summary = "A callable routine (either kernel or a free function)";
  // todo extend description with samples when dialect is formed.
  let description = [{
    TBD
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"OpBuilder& builder, OperationState &result, StringRef name, FunctionType type, "
              "ArrayRef<NamedAttribute> attrs = {}">
  ];

  let extraClassDeclaration = [{
    friend class OpTrait::FunctionLike<FuncOp>;
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }
    static StringRef getKernelAttributeName() {
      return "kernel";
    }
  }];

  // todo custom verifier and custom parser
}

def Compute_ReturnOp : Compute_Op<"return", [HasParent<"FuncOp">, NoSideEffect, Terminator]> {
  let summary = "Terminator for Compute functions";
  let description = [{
    TBD
  }];

  // todo custom printer, parser, verifier
}

//===----------------------------------------------------------------------===//
// Dimension operations
//===----------------------------------------------------------------------===//

def Compute_GlobalIdOp : Compute_Op<"global_id", [NoSideEffect]> {
  let summary = "Return global id of current work-item in ND-range";
  let description = [{
    TBD
  }];

  let arguments = (ins IndexAttr:$dim);
  let results = (outs Index);
}

def Compute_GlobalSizeOp : Compute_Op<"global_size", [NoSideEffect]> {
  let summary = "Return global size of dimension in ND-range";
  let description = [{
    TBD
  }];

  let arguments = (ins IndexAttr:$dim);
  let results = (outs Index);
}

def Compute_LocalIdOp : Compute_Op<"local_id", [NoSideEffect]> {
  let summary = "Return local id of current work-item in work-group";
  let description = [{
    TBD
  }];

  let arguments = (ins IndexAttr:$dim);
  let results = (outs Index);
}

def Compute_LocalSizeOp : Compute_Op<"local_size", [NoSideEffect]> {
  let summary = "Return global size of dimension in work-group";
  let description = [{
    TBD
  }];

  let arguments = (ins IndexAttr:$dim);
  let results = (outs Index);
}
