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

#ifndef CHIRA_PARSER_PARSER
#define CHIRA_PARSER_PARSER

#include "chira/dialect/sexpr/SExprOps.h"
#include "chira/parser/Tokenizer.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/LogicalResult.h"

namespace chira::parser {

class Parser {
public:
  using Input = Tokenizer::Result;

  llvm::LogicalResult Parse();

  Parser(mlir::MLIRContext &ctx, Input input)
      : module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx))),
        builder(module.getBodyRegion()), input(std::move(input)) {
    ctx.loadDialect<sexpr::SExprDialect>();
  }

  mlir::ModuleOp Module() { return module; }

  static mlir::Location MergeLoc(mlir::Location start_loc,
                                 mlir::Location end_loc);

private:
  mlir::ModuleOp module;
  mlir::OpBuilder builder;
  Input input;
};

} // namespace chira::parser

#endif // CHIRA_PARSER_PARSER
