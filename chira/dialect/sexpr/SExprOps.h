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

#ifndef CHIRA_DIALECT_SEXPR_SEXPROPS
#define CHIRA_DIALECT_SEXPR_SEXPROPS

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

#include "chira/dialect/sexpr/SExprOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "chira/dialect/sexpr/SExprOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "chira/dialect/sexpr/SExprOps.h.inc"

#endif // CHIRA_DIALECT_SEXPR_SEXPROPS
