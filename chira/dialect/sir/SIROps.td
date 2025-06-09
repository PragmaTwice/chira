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

class SIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<SIR_Dialect, mnemonic, traits> {
  let extraClassDeclaration = "";
}

def SIR_NumOp : SIR_Op<"num"> {
  let summary = "construct a variable from a number literal";

  let arguments = (ins
    Builtin_FloatAttr:$num
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

def SIR_YieldOp : SIR_Op<"yield"> {
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
      Variadic<SIR_Var>:$args
    );

    let results = (outs
      SIR_Var:$res
    );

    let assemblyFormat = [{
      $callee `(` $args `)` attr-dict
    }];
}

def SIR_LambdaOp : SIR_Op<"lambda"> {
    let summary = "create a lambda with captured variables";

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

def SIR_HoleOp : SIR_Op<"hole"> {
    let summary = "represents an unknown (hole) variable";

    let arguments = (ins
      Builtin_StringAttr:$id
    );

    let results = (outs
      SIR_Var:$res
    );

    let assemblyFormat = [{
      attr-dict
    }];
}

#endif
