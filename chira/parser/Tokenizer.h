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

#ifndef CHIRA_PARSER_TOKENZIER
#define CHIRA_PARSER_TOKENZIER

#include "chira/parser/Input.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace chira::parser {

struct Token {
  enum Kind : char {
    IDENTIFER,
    NUMBER,
    STRING,
    EXPR_BEGIN,
    EXPR_END,
  };

  Kind kind;
  std::string val;
  mlir::Location loc;

  Token(Kind kind, std::string val, mlir::Location loc)
      : kind(kind), val(std::move(val)), loc(loc) {}

  Token(Kind kind, std::string val, mlir::MLIRContext &ctx)
      : Token(kind, std::move(val), mlir::UnknownLoc::get(&ctx)) {}
  Token(Kind kind, std::string val, llvm::StringRef filename, size_t line,
        size_t column, mlir::MLIRContext &ctx)
      : Token(kind, std::move(val),
              mlir::FileLineColLoc::get(&ctx, filename, line, column)) {}

  static std::string Quoted(const std::string &val) {
    std::string res;
    res.reserve(val.size() + 2);

    res += "\"";
    for (char c : val) {
      if (c == '\n')
        res += "\\n";
      else if (c == '\t')
        res += "\\t";
      else if (c == '\"')
        res += "\\\"";
      else if (c == '\t')
        res += "\\t";
      else if (c == '\r')
        res += "\\r";
      else if (c == '\\')
        res += "\\\\";
      else
        res += c;
    }
    res += "\"";

    return res;
  }

  std::string ToString() {
    std::string str;
    llvm::raw_string_ostream os(str);
    os << "<";
    if (kind == EXPR_BEGIN)
      os << "BEGIN";
    else if (kind == EXPR_END)
      os << "END";
    else if (kind == IDENTIFER)
      os << "ID " << val;
    else if (kind == NUMBER)
      os << "NUM " << val;
    else if (kind == STRING)
      os << "STR " << Quoted(val);
    os << ", " << loc << ">";
    return str;
  }
};

class Tokenizer {
public:
  using Result = std::vector<Token>;

  Tokenizer(mlir::MLIRContext &ctx, Input input)
      : ctx(ctx), input(input.Content()), filename(input.Filename()) {}

  llvm::Error Tokenize();
  Result GetResult() { return result; }

private:
  char peek() const;
  char advance();
  bool eof() const;

  llvm::Error consumeExprBegin();
  llvm::Error consumeExprEnd();
  llvm::Error consumeString();
  llvm::Error consumeIdentifierOrNumber();
  llvm::Error consumeWhitespace();
  llvm::Error consumeComment();

  static bool isNumber(const std::string &value);

  mlir::MLIRContext &ctx;

  std::string input;
  std::string filename;

  size_t pos = 0;
  size_t column = 1;
  size_t line = 1;
  int paren_level = 0;

  Result result;
};

} // namespace chira::parser

#endif // CHIRA_PARSER_TOKENZIER
