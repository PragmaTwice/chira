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

#include "chira/parser/Tokenizer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/LogicalResult.h"

namespace chira::parser {

std::string Token::Quoted(const std::string &val) {
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

std::string Token::ToString() const {
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

char Tokenizer::peek() const {
  if (pos >= input.size())
    return 0;
  return input[pos];
}

char Tokenizer::advance() {
  char ch = input[pos++];
  if (ch == '\n') {
    line++;
    column = 1;
  } else {
    column++;
  }
  return ch;
}

bool Tokenizer::eof() const { return pos >= input.size(); }

llvm::LogicalResult Tokenizer::Tokenize() {
  while (!eof()) {
    char ch = peek();

    llvm::LogicalResult res = llvm::success();
    if (ch == '(') {
      res = consumeExprBegin();
    } else if (ch == ')') {
      res = consumeExprEnd();
    } else if (std::isspace(ch)) {
      res = consumeWhitespace();
    } else if (ch == ';') {
      res = consumeComment();
    } else if (ch == '"') {
      res = consumeString();
    } else {
      res = consumeIdentifierOrNumber();
    }

    if (llvm::failed(res))
      return res;
  }

  if (paren_level != 0) {
    emitError() << "parenthesis not matched";
    return llvm::failure();
  }

  return llvm::success();
}

mlir::FileLineColLoc Tokenizer::currentLoc() {
  return mlir::FileLineColLoc::get(&ctx, filename, line, column);
}

mlir::InFlightDiagnostic Tokenizer::emitError() {
  return diag.emit(currentLoc(), mlir::DiagnosticSeverity::Error);
}

llvm::LogicalResult Tokenizer::consumeExprBegin() {
  paren_level++;

  result.emplace_back(Token::EXPR_BEGIN, "", filename, line, column, ctx);

  advance();
  return llvm::success();
}

llvm::LogicalResult Tokenizer::consumeExprEnd() {
  paren_level--;

  if (paren_level < 0) {
    emitError() << "parenthesis not matched";
    return llvm::failure();
  }

  result.emplace_back(Token::EXPR_END, "", filename, line, column, ctx);

  advance();
  return llvm::success();
}

llvm::LogicalResult Tokenizer::consumeString() {
  size_t start_line = line;
  size_t start_col = column;
  std::string value;

  advance();

  while (!eof()) {
    char ch = advance();
    if (ch == '"') {
      result.emplace_back(Token::STRING, value, filename, start_line, start_col,
                          ctx);

      return llvm::success();
    } else if (ch == '\\') {
      if (eof()) {
        emitError() << "unterminated escape sequence";
        return llvm::failure();
      }

      char next = advance();
      if (next == 'n')
        value += '\n';
      else if (next == 't')
        value += '\t';
      else if (next == '\\')
        value += '\\';
      else if (next == '"')
        value += '"';
      else if (next == 'r')
        value += '\r';
      else {
        emitError() << "invalid escape character";
        return llvm::failure();
      }
    } else {
      value += ch;
    }
  }

  emitError() << "unterminated string literal";
  return llvm::failure();
}

llvm::LogicalResult Tokenizer::consumeIdentifierOrNumber() {
  size_t start_col = column;
  std::string value;

  while (!eof() && !isspace(peek()) && peek() != '(' && peek() != ')' &&
         peek() != ';' && peek() != '"') {
    value += advance();
  }

  bool is_number = isNumber(value);

  result.emplace_back(is_number ? Token::NUMBER : Token::IDENTIFER, value,
                      filename, line, start_col, ctx);
  return llvm::success();
}

llvm::LogicalResult Tokenizer::consumeWhitespace() {
  while (!eof() && isspace(peek()))
    advance();

  return llvm::success();
}

llvm::LogicalResult Tokenizer::consumeComment() {
  while (!eof() && peek() != '\n')
    advance();

  return llvm::success();
}

bool Tokenizer::isNumber(const std::string &value) {
  size_t i = 0, n = value.size();

  if (i < n && (value[i] == '+' || value[i] == '-'))
    i++;

  bool is_num = false;
  bool has_dot = false;
  bool has_exp = false;

  while (i < n) {
    if (std::isdigit(value[i])) {
      is_num = true;
      i++;
    } else if (value[i] == '.') {
      if (has_dot || has_exp)
        return false;
      has_dot = true;
      i++;
    } else if (value[i] == 'e' || value[i] == 'E') {
      if (!is_num || has_exp)
        return false;
      has_exp = true;
      is_num = false;
      i++;
      if (i < n && (value[i] == '+' || value[i] == '-'))
        i++;
    } else {
      break;
    }
  }

  return is_num && i == n;
}

} // namespace chira::parser
