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

struct Printer {
  static std::string Print(mlir::Value op);
  static std::string Print(mlir::ModuleOp op);
  static std::string Print(RootOp op);
  static std::string Print(SOp op);
  static std::string Print(IdOp op);
  static std::string Print(NumOp op);
  static std::string Print(StrOp op);
};

} // namespace chira::sexpr

#endif // CHIRA_DIALECT_SEXPR_SEXPRTOSTRING
