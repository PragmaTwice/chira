// Copyright 2025 PragmaTwice
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CHIR_OPS
#define CHIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "chira/dialect/sir/SIRDialect.td"

def CHIR_Dialect : Dialect {
  let name = "chir";
  let summary = "a middle-level SSA-based Chira IR";
  let cppNamespace = "::chira::chir";

  let useDefaultTypePrinterParser = 1;
}

def CHIR_Env : TypeDef<CHIR_Dialect, "Env"> {
  let mnemonic = "env";
  let summary = "capture environment in lambdas";

  let parameters = (ins
    "size_t":$size
  );

  let assemblyFormat = [{
    `<` $size `>`
  }];
}

def CHIR_Args : TypeDef<CHIR_Dialect, "Args"> {
  let mnemonic = "args";
  let summary = "arguments in lambdas";

  let parameters = (ins
    "size_t":$size
  );

  let assemblyFormat = [{
    `<` $size `>`
  }];
}

class CHIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<CHIR_Dialect, mnemonic, traits> {
  let extraClassDeclaration = "";
}

def CHIR_FuncRefOp : CHIR_Op<"func_ref"> {
  let summary = "a reference to a function symbol";

  let arguments = (ins
    Builtin_SymbolRefAttr:$func
  );

  let results = (outs
    SIR_Lambda:$res
  );

  let assemblyFormat = [{
    type($res) $func attr-dict
  }];
}

def CHIR_FuncOp : CHIR_Op<"func", [Symbol]> {
  let summary = "define a function symbol with its code body";

  let arguments = (ins
    Builtin_StringAttr:$sym_name,
    Builtin_IntegerAttr:$cap_size
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = [{
    $sym_name $body attr-dict
  }];
}

def CHIR_EnvLoadOp : CHIR_Op<"env_load"> {
  let summary = "load a variable from the environment";

  let arguments = (ins
    CHIR_Env:$env,
    Builtin_IntegerAttr:$index
  );

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    type($env) $env `[` $index `]` attr-dict
  }];
}

def CHIR_EnvStoreOp : CHIR_Op<"env_store"> {
  let summary = "store a variable into the environment";

  let arguments = (ins
    CHIR_Env:$env,
    Builtin_IntegerAttr:$index,
    SIR_Var:$var
  );

  let assemblyFormat = [{
    type($env) $env `[` $index `]` `=` $var attr-dict
  }];
}

def CHIR_EnvOp : CHIR_Op<"env"> {
  let summary = "create a new environment";

  let results = (outs
    CHIR_Env:$env
  );

  let assemblyFormat = [{
    type($env) attr-dict
  }];
}

def CHIR_ClosureOp : CHIR_Op<"closure"> {
  let summary = "create a new closure from lambda and env";

  let arguments = (ins
    SIR_Lambda:$lambda,
    CHIR_Env:$env
  );

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    type($lambda) $lambda type($env) $env attr-dict
  }];
}

def CHIR_AsBoolOp : CHIR_Op<"as_bool"> {
  let summary = "convert a variable to a boolean for conditional branching";

  let arguments = (ins
    SIR_Var:$var
  );

  let results = (outs
    I1:$res
  );

  let assemblyFormat = [{
    $var attr-dict
  }];
}

def CHIR_ArgsLoadOp : CHIR_Op<"args_load"> {
  let summary = "load a variable from arguments";

  let arguments = (ins
    CHIR_Args:$args,
    Builtin_IntegerAttr:$index
  );

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    type($args) $args `[` $index `]` attr-dict
  }];
}

#endif
