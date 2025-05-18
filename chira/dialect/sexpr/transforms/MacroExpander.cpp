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

#include "chira/dialect/sexpr/SExprOps.h"
#include "chira/dialect/sexpr/transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"

namespace chira::sexpr {

namespace {

struct MacroExpanderPass
    : public mlir::PassWrapper<MacroExpanderPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    module->walk([&](DefineSyntaxOp ds) {
      bool success = true;
      for (auto &i : ds->getUses()) {
        std::set<mlir::Operation *> visited;

        auto op = i.getOwner();
        for (size_t j = i.getOperandNumber(); j < op->getNumOperands(); j++) {
          auto expr = op->getOperand(j);
          if (!visit(expr.getDefiningOp(), ds, visited)) {
            success = false;
            signalPassFailure();
          }
        }
      }

      if (success) {
        for (auto &i : ds->getUses()) {
          auto owner = i.getOwner();

          std::vector<mlir::Value> operands{owner->getOperands().begin(),
                                            owner->getOperands().end()};
          operands.erase(operands.begin() + i.getOperandNumber());
          owner->setOperands(operands);
        }
        ds.erase();
      }
    });
  }

  bool visit(mlir::Operation *op, DefineSyntaxOp ds,
             std::set<mlir::Operation *> &visited) {
    auto expr = llvm::dyn_cast<SOp>(op);
    if (!expr || visited.count(op))
      return true;

    visited.insert(op);

    bool success = true;
    for (auto i : expr.getExprs())
      success &= visit(i.getDefiningOp(), ds, visited);

    if (expr->getOperands().empty())
      return success;

    auto first = llvm::dyn_cast<IdOp>(expr->getOperand(0).getDefiningOp());
    if (!first)
      return success;

    if (first.getId() !=
        llvm::dyn_cast<IdOp>(ds.getName().getDefiningOp()).getId())
      return success;

    return success & match(expr, ds);
  }

  bool match(SOp op, DefineSyntaxOp ds) {
    std::set<llvm::StringRef> literals;
    for (auto e :
         llvm::dyn_cast<SOp>(ds.getLiterals().getDefiningOp()).getExprs()) {
      auto id = llvm::dyn_cast<IdOp>(e.getDefiningOp());
      literals.insert(id.getId());
    }
    literals.insert(llvm::dyn_cast<IdOp>(ds.getName().getDefiningOp()).getId());

    std::map<llvm::StringRef, mlir::Operation *> bindings;
    bool matched = false;

    for (auto p : ds.getPatterns()) {
      auto pop = llvm::dyn_cast<SOp>(p.getDefiningOp());
      auto pat = pop.getExprs()[0].getDefiningOp();
      auto temp = pop.getExprs()[1].getDefiningOp();

      if (match(op, pat, literals, bindings)) {
        matched = true;

        mlir::OpBuilder builder(op);
        auto new_op = construct(builder, temp, bindings);

        op.replaceAllUsesWith(new_op);
        op.erase();
        break;
      }
    }

    if (!matched) {
      emitError(op.getLoc()) << "no matching pattern found for expansion";
      return false;
    }

    return true;
  }

  bool match(mlir::Operation *actual, mlir::Operation *expected,
             const std::set<llvm::StringRef> &literals,
             std::map<llvm::StringRef, mlir::Operation *> &bindings) {
    if (auto e = llvm::dyn_cast<SOp>(expected)) {
      auto a = llvm::dyn_cast<SOp>(actual);
      if (!a || e.getExprs().size() != a.getExprs().size())
        return false;

      for (size_t i = 0; i < e.getExprs().size(); i++) {
        auto expected_expr = e.getExprs()[i].getDefiningOp();
        auto actual_expr = a.getExprs()[i].getDefiningOp();

        if (!match(actual_expr, expected_expr, literals, bindings)) {
          return false;
        }
      }

      return true;
    } else if (auto id = llvm::dyn_cast<IdOp>(expected)) {
      auto id_name = id.getId();
      if (literals.count(id_name)) {
        auto actual_id = llvm::dyn_cast<IdOp>(actual);
        return actual_id && actual_id.getId() == id_name;
      } else {
        bindings[id_name] = actual;
        return true;
      }
    } else if (auto num = llvm::dyn_cast<NumOp>(expected)) {
      auto actual_num = llvm::dyn_cast<NumOp>(actual);
      return actual_num && actual_num.getNum() == num.getNum();
    } else if (auto str = llvm::dyn_cast<StrOp>(expected)) {
      auto actual_str = llvm::dyn_cast<StrOp>(actual);
      return actual_str && actual_str.getStr() == str.getStr();
    }

    return false;
  }

  mlir::Operation *
  construct(mlir::OpBuilder &builder, mlir::Operation *temp,
            const std::map<llvm::StringRef, mlir::Operation *> &bindings) {
    auto expr_type = ExprType::get(temp->getContext());
    if (auto expr = llvm::dyn_cast<SOp>(temp)) {
      std::vector<mlir::Value> new_operands;
      for (auto i : expr.getExprs()) {
        new_operands.push_back(
            construct(builder, i.getDefiningOp(), bindings)->getResult(0));
      }

      return builder.create<SOp>(temp->getLoc(), expr_type, new_operands);
    } else if (auto id = llvm::dyn_cast<IdOp>(temp)) {
      auto it = bindings.find(id.getId());
      if (it != bindings.end()) {
        return it->second;
      } else {
        return builder.create<IdOp>(temp->getLoc(), expr_type, id.getId());
      }
    } else if (auto num = llvm::dyn_cast<NumOp>(temp)) {
      return builder.create<NumOp>(temp->getLoc(), expr_type, num.getNum());
    } else if (auto str = llvm::dyn_cast<StrOp>(temp)) {
      return builder.create<StrOp>(temp->getLoc(), expr_type, str.getStr());
    }

    llvm_unreachable("unexpected operation type");
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createMacroExpanderPass() {
  return std::make_unique<MacroExpanderPass>();
}

} // namespace chira::sexpr
