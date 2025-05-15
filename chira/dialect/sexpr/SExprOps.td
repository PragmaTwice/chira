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

#ifndef SEXPR_OPS
#define SEXPR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SExpr_Dialect : Dialect {
  let name = "sexpr";
  let summary = "S-expression Syntax Representation";
  let cppNamespace = "::chira::sexpr";

  let useDefaultTypePrinterParser = 1;
}

def SExpr_Expr : TypeDef<SExpr_Dialect, "Expr"> {
  let mnemonic = "expr";
  let summary = "expression placeholder in SExpr dialect";
}

class SExpr_Op<string mnemonic, list<Trait> traits = []> :
    Op<SExpr_Dialect, mnemonic, traits> {
  let extraClassDeclaration = "";
}

def SExpr_SOp : SExpr_Op<"s", [Pure]> {
  let summary = "represents an S-expression";

  let arguments = (ins
    Variadic<SExpr_Expr>:$exprs
  );

  let results = (outs
    SExpr_Expr:$res
  );

  let assemblyFormat = [{
    $exprs attr-dict
  }];
}

def SExpr_RootOp : SExpr_Op<"root"> {
  let summary = "represents the root node of the S-expression tree";

  let arguments = (ins
    Variadic<SExpr_Expr>:$exprs
  );

  let results = (outs
    SExpr_Expr:$res
  );

  let assemblyFormat = [{
    $exprs attr-dict
  }];
}

def SExpr_IdOp : SExpr_Op<"id", [Pure]> {
  let summary = "represents an identifier";

  let arguments = (ins
    Builtin_StringAttr:$id
  );

  let results = (outs
    SExpr_Expr:$res
  );

  let assemblyFormat = [{
    $id attr-dict
  }];
}

def SExpr_NumOp : SExpr_Op<"num", [Pure]> {
  let summary = "represents a number literal";

  let arguments = (ins
    Builtin_StringAttr:$num
  );

  let results = (outs
    SExpr_Expr:$res
  );

  let assemblyFormat = [{
    $num attr-dict
  }];
}

def SExpr_StrOp : SExpr_Op<"str", [Pure]> {
  let summary = "represents a string literal";

  let arguments = (ins
    Builtin_StringAttr:$str
  );

  let results = (outs
    SExpr_Expr:$res
  );

  let assemblyFormat = [{
    $str attr-dict
  }];
}

def SExpr_DefineSyntaxOp : SExpr_Op<"define_syntax"> {
  let summary = "represents a (define-syntax ...) in Scheme";

  let arguments = (ins
    SExpr_Expr:$name,
    SExpr_Expr:$literals,
    Variadic<SExpr_Expr>:$patterns
  );

  let results = (outs
    SExpr_Expr:$res
  );

  let assemblyFormat = [{
    $name $literals $patterns attr-dict
  }];
}

#endif // SEXPR_OPS
