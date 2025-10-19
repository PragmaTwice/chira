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

#ifndef SIR_DIALECT
#define SIR_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def SIR_Dialect : Dialect {
  let name = "sir";
  let summary = "a high-level ANF-based Scheme IR";
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

  let parameters = (ins
    // param_size + cap_size = size of block arguments,
    // e.g. for `sir.lambda { x, y, z: ... } cap_size=1`,
    // params are `x` and `y`, and `z` is a captured variable
    "size_t":$param_size,
    "size_t":$cap_size
  );

  let assemblyFormat = [{
    `<` $param_size `,` $cap_size `>`
  }];
}

#endif
