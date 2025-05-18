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

#include "chira/dialect/sexpr/SExprToString.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace chira::sexpr {

std::string ToString(mlir::Value op) {
  if (auto id = llvm::dyn_cast<IdOp>(op.getDefiningOp())) {
    return ToString(id);
  } else if (auto num = llvm::dyn_cast<NumOp>(op.getDefiningOp())) {
    return ToString(num);
  } else if (auto str = llvm::dyn_cast<StrOp>(op.getDefiningOp())) {
    return ToString(str);
  } else if (auto s = llvm::dyn_cast<SOp>(op.getDefiningOp())) {
    return ToString(s);
  } else if (auto root = llvm::dyn_cast<RootOp>(op.getDefiningOp())) {
    return ToString(root);
  }

  llvm_unreachable("unexpected operation type");
}

std::string ToString(mlir::ModuleOp op) {
  std::string result;
  op->walk([&](RootOp op) { result += ToString(op); });
  return result;
}

std::string ToString(RootOp op) {
  std::string result;
  for (auto expr : op.getExprs()) {
    result += ToString(expr) + "\n";
  }
  return result;
}

std::string ToString(SOp op) {
  std::string result;
  result += "(";
  bool first = true;
  for (auto expr : op.getExprs()) {
    if (!first) {
      result += " ";
    } else {
      first = false;
    }
    result += ToString(expr);
  }
  result += ")";
  return result;
}

std::string ToString(IdOp op) { return op.getId().getValue().str(); }

std::string ToString(NumOp op) { return op.getNum().getValue().str(); }

std::string ToString(StrOp op) {
  std::string str;
  llvm::raw_string_ostream os(str);
  op.getStr().print(os);
  return os.str();
}

} // namespace chira::sexpr
