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

#ifndef CHIRA_DIALECT_SEXPR_SEXPRTOSTRING
#define CHIRA_DIALECT_SEXPR_SEXPRTOSTRING

#include "chira/dialect/sexpr/SExprOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace chira::sexpr {

std::string ToString(mlir::Value op);
std::string ToString(mlir::ModuleOp op);
std::string ToString(RootOp op);
std::string ToString(SOp op);
std::string ToString(IdOp op);
std::string ToString(NumOp op);
std::string ToString(StrOp op);

} // namespace chira::sexpr

#endif // CHIRA_DIALECT_SEXPR_SEXPRTOSTRING
