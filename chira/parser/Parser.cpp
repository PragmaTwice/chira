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
#include "mlir/IR/Location.h"
#include "llvm/Support/Error.h"

namespace chira::parser {

llvm::Error Parser::Parse() {
  std::vector<std::vector<mlir::Value>> stack;
  std::vector<mlir::Value> current;

  auto ctx = builder.getContext();
  auto type = sexpr::ExprType::get(ctx);
  auto unknown_loc = mlir::UnknownLoc::get(ctx);

  for (const auto &token : input) {
    if (token.kind == Token::EXPR_BEGIN) {
      stack.push_back(current);
      current.clear();
    } else if (token.kind == Token::EXPR_END) {
      auto expr = current;
      current = stack.back();
      stack.pop_back();
      current.push_back(builder.create<sexpr::SOp>(token.loc, type, expr));
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

  return llvm::Error::success();
}

} // namespace chira::parser
