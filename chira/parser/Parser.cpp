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

#include "chira/parser/Parser.h"
#include "chira/dialect/sexpr/SExprOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/LogicalResult.h"

namespace chira::parser {

mlir::Location Parser::MergeLoc(mlir::Location start_loc,
                                mlir::Location end_loc) {
  auto start =
      mlir::dyn_cast<mlir::FileLineColLoc>((mlir::LocationAttr)start_loc);

  auto end = mlir::dyn_cast<mlir::FileLineColLoc>((mlir::LocationAttr)end_loc);

  return mlir::Location(mlir::FileLineColRange::get(
      start.getContext(), start.getFilename(), start.getLine(),
      start.getColumn(), end.getLine(), end.getColumn()));
}

llvm::LogicalResult Parser::Parse() {
  std::vector<std::pair<std::vector<mlir::Value>, mlir::Location>> stack;
  std::vector<mlir::Value> current;

  auto ctx = builder.getContext();
  auto type = sexpr::ExprType::get(ctx);
  auto unknown_loc = mlir::UnknownLoc::get(ctx);

  for (const auto &token : input) {
    if (token.kind == Token::EXPR_BEGIN) {
      stack.emplace_back(current, token.loc);
      current.clear();
    } else if (token.kind == Token::EXPR_END) {
      auto expr = current;
      mlir::Location start_loc = unknown_loc;
      std::tie(current, start_loc) = stack.back();
      stack.pop_back();
      current.push_back(builder.create<sexpr::SOp>(
          MergeLoc(start_loc, token.loc), type, expr));
    } else if (token.kind == Token::IDENTIFER) {
      current.push_back(builder.create<sexpr::IdOp>(
          token.loc, type, mlir::StringAttr::get(ctx, token.val)));
    } else if (token.kind == Token::NUMBER) {
      current.push_back(builder.create<sexpr::NumOp>(
          token.loc, type, mlir::StringAttr::get(ctx, token.val)));
    } else if (token.kind == Token::STRING) {
      current.push_back(builder.create<sexpr::StrOp>(
          token.loc, type, mlir::StringAttr::get(ctx, token.val)));
    }
  }

  builder.create<sexpr::RootOp>(unknown_loc, type, current);

  if (failed(mlir::verify(module))) {
    mlir::emitError(unknown_loc)
        << "the generated IR failed to pass verification";
    return llvm::failure();
  }

  return llvm::success();
}

} // namespace chira::parser
