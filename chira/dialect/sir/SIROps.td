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

#ifndef SIR_OPS
#define SIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def SIR_Dialect : Dialect {
  let name = "sir";
  let summary = "Scheme ANF-based high-level IR";
  let cppNamespace = "::chira::sir";

  let useDefaultTypePrinterParser = 1;
}

def SIR_Var : TypeDef<SIR_Dialect, "Var"> {
  let mnemonic = "var";
  let summary = "variable in SIR dialect";
}

// lambda should be binded to form a variable,
// i.e. bind(lambda, caps..) -> var
def SIR_Lambda : TypeDef<SIR_Dialect, "Lambda"> {
  let mnemonic = "lambda";
  let summary = "lambda in SIR dialect";
}

class SIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<SIR_Dialect, mnemonic, traits> {
  let extraClassDeclaration = "";
}

def SIR_NumOp : SIR_Op<"num"> {
  let summary = "construct a variable from a number literal";

  let arguments = (ins
    AnyAttrOf<[Builtin_FloatAttr, Builtin_IntegerAttr]>:$num
  );

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    $num attr-dict
  }];
}

def SIR_StrOp : SIR_Op<"str"> {
  let summary = "construct a variable from a string literal";

  let arguments = (ins
    Builtin_StringAttr:$str
  );

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    $str attr-dict
  }];
}

def SIR_YieldOp : SIR_Op<"yield", [Terminator]> {
  let summary = "yield a variable and terminate the current block";

  let arguments = (ins
    SIR_Var:$var
  );

  let assemblyFormat = [{
    $var attr-dict
  }];
}

def SIR_IfOp : SIR_Op<"if"> {
  let summary = "conditional branching";

  let arguments = (ins
    SIR_Var:$cond
  );

  let results = (outs
    SIR_Var:$res
  );

  let regions = (region
    SizedRegion<1>:$then,
    SizedRegion<1>:$else
  );

  let assemblyFormat = [{
    $cond $then `else` $else attr-dict
  }];
}

def SIR_CallOp : SIR_Op<"call"> {
  let summary = "call a function with arguments";

  let arguments = (ins
    SIR_Var:$callee,
    Variadic<SIR_Var>:$args,
    OptionalAttr<Builtin_UnitAttr>:$tail
  );

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict
  }];
}

// `%f = sir.closure { ... } %x %y` equals to
// ```
//  %f_l = sir.lambda { ... } cap_size=2
//  %f = sir.bind %f_l (%x %y)
// ```
// this op cannot be used for recursive functions,
// use `sir.lambda` and `sir.bind` instead
def SIR_ClosureOp : SIR_Op<"closure"> {
  let summary = "create a closure with an in-place block and captures";

  let arguments = (ins
    Variadic<SIR_Var>:$caps
  );

  let results = (outs
    SIR_Var:$res
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = [{
    $caps $body attr-dict
  }];
}

def SIR_LambdaOp : SIR_Op<"lambda"> {
  let summary = "create a lambda";

  let arguments = (ins
    // param_size + cap_size = size of block arguments,
    // e.g. for `sir.lambda { x, y, z: ... } cap_size=1`,
    // params are `x` and `y`, and `z` is a captured variable
    Builtin_IntegerAttr:$cap_size
  );

  let results = (outs
    SIR_Lambda:$res
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = [{
    $body attr-dict
  }];
}

def SIR_BindOp : SIR_Op<"bind", [AttrSizedOperandSegments]> {
  let summary = "create a closure from lambdas and captures";

  let arguments = (ins
    Variadic<SIR_Lambda>:$lambdas,
    // lambda in captures means recursive functions,
    // and all lambdas in $caps must be present in $lambdas
    // e.g. sir.bind f f -> recursive
    //      sir.bind f x -> non-recursive
    Variadic<AnyTypeOf<[SIR_Var, SIR_Lambda]>>:$caps
  );

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    $lambdas type($caps) $caps attr-dict
  }];
}

def SIR_ArithPrimOp : SIR_Op<"arith_prim"> {
  let summary = "call an arithmetic primitive operation with arguments";

  let arguments = (ins
    Builtin_StringAttr:$op,
    Variadic<SIR_Var>:$args
  );

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    $op `(` $args `)` attr-dict
  }];
}

def SIR_IOPrimOp : SIR_Op<"io_prim"> {
  let summary = "call an IO primitive operation with arguments";

  let arguments = (ins
    Builtin_StringAttr:$op,
    Variadic<SIR_Var>:$args
  );

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    $op `(` $args `)` attr-dict
  }];
}

def SIR_UnspecifiedOp : SIR_Op<"unspec"> {
  let summary = "construct a variable with unspecified value";

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    attr-dict
  }];
}

def SIR_SetOp : SIR_Op<"set"> {
  let summary = "set a variable to a new value";

  let arguments = (ins
    SIR_Var:$var,
    SIR_Var:$value
  );

  let assemblyFormat = [{
    $var `=` $value attr-dict
  }];
}

def SIR_UnresolvedNameOp : SIR_Op<"unresolved_name"> {
  let summary = "a placeholder variable for an unresolved name";

  let arguments = (ins
    Builtin_StringAttr:$name
  );

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    $name attr-dict
  }];
}

// low-level operations (for MLIR dialect lowering)
// TODO: should be moved to a separate dialect like "chir"

def SIR_FuncRefOp : SIR_Op<"func_ref"> {
  let summary = "a reference to a function symbol";

  let arguments = (ins
    Builtin_SymbolRefAttr:$func
  );

  let results = (outs
    SIR_Lambda:$res
  );

  let assemblyFormat = [{
    $func attr-dict
  }];
}

// `%f = sir.bind_ref f x y` is equivalent to
// ```
//  %f_l = sir.func_ref @f
//  %f = sir.bind %f_l x y
// ```
// and it cannot be used for recursive functions
def SIR_BindRefOp : SIR_Op<"bind_ref"> {
  let summary = "bind a function symbol with its captures";

  let arguments = (ins
    Builtin_SymbolRefAttr:$func,
    Variadic<SIR_Var>:$caps
  );

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    $func $caps attr-dict
  }];
}

def SIR_FuncOp : SIR_Op<"func", [Symbol]> {
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

def SIR_Env : TypeDef<SIR_Dialect, "Env"> {
  let mnemonic = "env";
  let summary = "capture environment in lambdas";

  let parameters = (ins
    "size_t":$size
  );

  let assemblyFormat = [{
    `<` $size `>`
  }];
}

def SIR_EnvLoadOp : SIR_Op<"env_load"> {
  let summary = "load a variable from the environment";

  let arguments = (ins
    SIR_Env:$env,
    Builtin_IntegerAttr:$index
  );

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    type($env) $env `[` $index `]` attr-dict
  }];
}

def SIR_EnvStoreOp : SIR_Op<"env_store"> {
  let summary = "store a variable into the environment";

  let arguments = (ins
    SIR_Env:$env,
    Builtin_IntegerAttr:$index,
    SIR_Var:$var
  );

  let assemblyFormat = [{
    type($env) $env `[` $index `]` `=` $var attr-dict
  }];
}

def SIR_EnvOp : SIR_Op<"env"> {
  let summary = "create a new environment";

  let results = (outs
    SIR_Env:$env
  );

  let assemblyFormat = [{
    type($env) attr-dict
  }];
}

def SIR_ClosureFromEnvOp : SIR_Op<"closure_from_env"> {
  let summary = "create a new closure from lambda and env";

  let arguments = (ins
    SIR_Lambda:$lambda,
    SIR_Env:$env
  );

  let results = (outs
    SIR_Var:$res
  );

  let assemblyFormat = [{
    $lambda type($env) $env attr-dict
  }];
}

def SIR_AsBoolOp : SIR_Op<"as_bool"> {
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

#endif
