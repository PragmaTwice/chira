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

#include "chira/dialect/sexpr/SExprPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace chira::sexpr {

std::string Printer::Print(mlir::Value op, size_t indent,
                           size_t max_line_length) {
  if (auto id = llvm::dyn_cast<IdOp>(op.getDefiningOp())) {
    return Print(id);
  } else if (auto num = llvm::dyn_cast<NumOp>(op.getDefiningOp())) {
    return Print(num);
  } else if (auto str = llvm::dyn_cast<StrOp>(op.getDefiningOp())) {
    return Print(str);
  } else if (auto s = llvm::dyn_cast<SOp>(op.getDefiningOp())) {
    return Print(s, indent, max_line_length);
  } else if (auto root = llvm::dyn_cast<RootOp>(op.getDefiningOp())) {
    return Print(root, indent, max_line_length);
  }

  llvm_unreachable("unexpected operation type");
}

std::string Printer::Print(mlir::ModuleOp op, size_t indent,
                           size_t max_line_length) {
  std::string result;
  op->walk([&](RootOp op) { result += Print(op, indent, max_line_length); });
  return result;
}

std::string Printer::Print(RootOp op, size_t indent, size_t max_line_length) {
  std::string result;
  for (auto expr : op.getExprs()) {
    result += Print(expr, indent, max_line_length) + "\n";
  }
  return result;
}

std::string Printer::IndentStr(size_t indent) {
  return std::string(indent, ' ');
}

std::string Printer::Print(SOp op, size_t indent, size_t max_line_length) {
  std::string one_line = "(";

  bool first = true;
  for (auto expr : op.getExprs()) {
    if (!first) {
      one_line += " ";
    } else {
      first = false;
    }

    one_line += Print(expr, 0, max_line_length);
  }
  one_line += ")";

  if (one_line.length() + indent <= max_line_length) {
    return IndentStr(indent) + one_line;
  }

  std::string result = IndentStr(indent) + "(\n";

  bool firstLine = true;
  for (auto expr : op.getExprs()) {
    if (!firstLine) {
      result += "\n";
    } else {
      firstLine = false;
    }

    if (!llvm::isa<SOp>(expr.getDefiningOp())) {
      result +=
          IndentStr(indent + INDENT_WIDTH) + Print(expr, 0, max_line_length);
    } else {
      result += Print(expr, indent + INDENT_WIDTH, max_line_length);
    }
  }

  result += "\n" + IndentStr(indent) + ")";
  return result;
}

std::string Printer::Print(IdOp op) { return op.getId().getValue().str(); }

std::string Printer::Print(NumOp op) { return op.getNum().getValue().str(); }

std::string Printer::Print(StrOp op) {
  std::string str;
  llvm::raw_string_ostream os(str);
  op.getStr().print(os);
  return os.str();
}

} // namespace chira::sexpr
