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

#include "chira/conversion/sexprtosir/SExprToSIR.h"
#include "chira/dialect/sexpr/SExprOps.h"
#include "chira/dialect/sir/SIROps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace chira {

namespace {

struct SExprToSIRConversionPass
    : public mlir::PassWrapper<SExprToSIRConversionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();

    sexpr::RootOp root;
    module->walk([&](sexpr::RootOp op) { root = op; });

    auto new_module = mlir::ModuleOp::create(module->getLoc());
    mlir::OpBuilder builder(new_module.getRegion());

    bool success = true;
    std::map<llvm::StringRef, mlir::Value> global_scope;
    for (auto e : root.getExprs()) {
      if (!visitOp(e.getDefiningOp(), builder, global_scope)) {
        success = false;
      }
    }

    module.getRegion().takeBody(new_module.getRegion());
    if (!success) {
      signalPassFailure();
    }
  };

  mlir::Value visitOp(mlir::Operation *op, mlir::OpBuilder &builder,
                      std::map<llvm::StringRef, mlir::Value> &scope) {
    if (auto s = llvm::dyn_cast<sexpr::SOp>(op)) {
      return visitExpr(s, builder, scope);
    } else if (auto id = llvm::dyn_cast<sexpr::IdOp>(op)) {
      if (auto it = scope.find(id.getId().strref()); it != scope.end()) {
        return it->second;
      } else {
        mlir::emitError(id.getLoc()) << "undefined identifier " << id.getId();
        return {};
      }
    } else if (auto str = llvm::dyn_cast<sexpr::StrOp>(op)) {
      auto var_type = sir::VarType::get(&getContext());
      return builder.create<sir::StrOp>(op->getLoc(), var_type, str.getStr());
    } else if (auto num = llvm::dyn_cast<sexpr::NumOp>(op)) {
      auto var_type = sir::VarType::get(&getContext());
      auto val =
          llvm::APFloat(llvm::APFloat::IEEEquad(), num.getNum().getValue());
      auto attr =
          mlir::FloatAttr::get(mlir::Float128Type::get(&getContext()), val);
      return builder.create<sir::NumOp>(op->getLoc(), var_type, attr);
    }

    llvm_unreachable("unexpected operation type");
  }

  bool isPrimOp(llvm::StringRef id) {
    return id == "+" || id == "-" || id == "*" || id == "/" || id == "=" ||
           id == "<" || id == "<=" || id == ">" || id == ">=" ||
           id == "display" || id == "newline";
  }

  mlir::Value visitExpr(sexpr::SOp expr, mlir::OpBuilder &builder,
                        std::map<llvm::StringRef, mlir::Value> &scope) {
    auto exprs = expr.getExprs();
    if (exprs.empty()) {
      mlir::emitError(expr.getLoc())
          << "empty S-expression is not allowed here";
      return {};
    }

    auto first = exprs[0].getDefiningOp();
    if (auto opcode = llvm::dyn_cast<sexpr::IdOp>(first)) {
      if (opcode.getId() == "define") {
        return visitDefine(expr, builder, scope);
      } else if (isPrimOp(opcode.getId())) {
        return visitPrim(opcode.getId(), expr, builder, scope);
      } else if (opcode.getId() == "if") {
        return visitIf(expr, builder, scope);
      } else if (opcode.getId() == "begin") {
        return visitBegin(expr, builder, scope);
      } else if (opcode.getId() == "lambda") {
        return visitLambda(expr, builder, scope);
      }
    }

    return visitCall(expr, builder, scope);
  }

  mlir::Value visitBegin(sexpr::SOp expr, mlir::OpBuilder &builder,
                         std::map<llvm::StringRef, mlir::Value> &scope) {
    auto exprs = expr.getExprs();
    if (exprs.size() < 2) {
      mlir::emitError(expr.getLoc())
          << "expected more than 1 argument in (begin <expr>..)";
      return {};
    }

    auto new_scope = scope;

    mlir::Value last;
    for (auto e : exprs.drop_front(1)) {
      last = visitOp(e.getDefiningOp(), builder, new_scope);
      if (!last) {
        return {};
      }
    }

    return last;
  }

  mlir::Value visitIf(sexpr::SOp expr, mlir::OpBuilder &builder,
                      std::map<llvm::StringRef, mlir::Value> &scope) {
    auto exprs = expr.getExprs();
    if (exprs.size() < 3 || exprs.size() > 4) {
      mlir::emitError(expr.getLoc())
          << "expected 2 or 3 arguments in (if <cond> <then> [<else>])";
      return {};
    }

    auto var_type = sir::VarType::get(&getContext());
    auto cond = visitOp(exprs[1].getDefiningOp(), builder, scope);
    if (!cond) {
      return {};
    }
    auto if_op = builder.create<sir::IfOp>(expr->getLoc(), var_type, cond);

    auto ip = builder.saveInsertionPoint();
    auto then_block = &if_op.getThen().emplaceBlock();
    builder.setInsertionPointToStart(then_block);

    auto then = visitOp(exprs[2].getDefiningOp(), builder, scope);
    if (!then) {
      return {};
    }

    builder.create<sir::YieldOp>(exprs[2].getLoc(), then);

    auto else_block = &if_op.getElse().emplaceBlock();
    builder.setInsertionPointToStart(else_block);
    if (exprs.size() == 4) {
      auto else_val = visitOp(exprs[3].getDefiningOp(), builder, scope);
      if (!else_val) {
        return {};
      }
      builder.create<sir::YieldOp>(exprs[3].getLoc(), else_val);
    } else {
      auto unspec =
          builder.create<sir::UnspecifiedOp>(expr->getLoc(), var_type);
      builder.create<sir::YieldOp>(expr->getLoc(), unspec);
    }

    builder.restoreInsertionPoint(ip);
    return if_op;
  }

  mlir::Value visitCall(sexpr::SOp expr, mlir::OpBuilder &builder,
                        std::map<llvm::StringRef, mlir::Value> &scope) {
    auto exprs = expr.getExprs();

    auto first = expr.getExprs().front().getDefiningOp();
    auto func = visitOp(first, builder, scope);
    if (!func) {
      return {};
    }

    std::vector<mlir::Value> operands;
    for (auto e : exprs.drop_front(1)) {
      if (auto v = visitOp(e.getDefiningOp(), builder, scope)) {
        operands.push_back(v);
      } else {
        return {};
      }
    }

    auto var_type = sir::VarType::get(&getContext());
    return builder.create<sir::CallOp>(expr->getLoc(), var_type, func, operands,
                                       nullptr);
  }

  mlir::Value visitDefine(sexpr::SOp expr, mlir::OpBuilder &builder,
                          std::map<llvm::StringRef, mlir::Value> &scope) {
    auto exprs = expr.getExprs();
    if (exprs.size() < 3) {
      mlir::emitError(expr.getLoc())
          << "expected more than 2 arguments in (define <name> <value>..)";
      return {};
    }

    auto name_op = exprs[1].getDefiningOp();
    if (auto name = llvm::dyn_cast<sexpr::IdOp>(name_op)) {
      if (exprs.size() != 3) {
        mlir::emitError(expr.getLoc())
            << "expected 2 arguments in (define <name> <value>)";
        return {};
      }

      auto value = visitOp(exprs[2].getDefiningOp(), builder, scope);
      if (!value) {
        return {};
      }

      value.getDefiningOp()->setAttr("defined_name", name.getIdAttr());
      scope.insert_or_assign(name.getId().strref(), value);
      auto var_type = sir::VarType::get(&getContext());
      return builder.create<sir::UnspecifiedOp>(expr.getLoc(), var_type);
    } else if (auto names = llvm::dyn_cast<sexpr::SOp>(name_op)) {
      if (names.getExprs().empty()) {
        mlir::emitError(expr.getLoc()) << "<name> in (define <name> <value>) "
                                          "should not be an empty S-expression";
        return {};
      }

      std::vector<llvm::StringRef> name_and_args;
      for (auto e : names.getExprs()) {
        if (auto id = llvm::dyn_cast<sexpr::IdOp>(e.getDefiningOp())) {
          name_and_args.push_back(id.getId().strref());
        } else {
          mlir::emitError(name_op->getLoc())
              << "function name and arguments in (define ..) should be an "
                 "identifier";
          return {};
        }
      }

      auto name = name_and_args.front();
      auto args = llvm::ArrayRef(name_and_args).drop_front();

      auto var_type = sir::VarType::get(&getContext());
      auto lambda = builder.create<sir::ClosureOp>(expr->getLoc(), var_type,
                                                   mlir::ValueRange{});
      lambda->setAttr("defined_name",
                      mlir::StringAttr::get(&getContext(), name));
      scope.insert_or_assign(name, lambda);

      auto ip = builder.saveInsertionPoint();

      auto block = &lambda.getBody().emplaceBlock();
      builder.setInsertionPointToStart(block);

      auto new_scope = scope;
      for (size_t i = 0; i < args.size(); ++i) {
        auto arg =
            block->addArgument(var_type, names.getExprs()[i + 1].getLoc());
        new_scope.insert_or_assign(args[i], arg);
      }

      mlir::Value ret;
      for (auto e : exprs.drop_front(2)) {
        if (auto v = visitOp(e.getDefiningOp(), builder, new_scope)) {
          ret = v;
        } else {
          return {};
        }
      }

      builder.create<sir::YieldOp>(exprs.back().getLoc(), ret);
      builder.restoreInsertionPoint(ip);

      return lambda;
    } else {
      mlir::emitError(name_op->getLoc())
          << "<name> in (define <name> <value>) should be an identifier or "
             "S-expression";
      return {};
    }
  }

  mlir::Value visitPrim(llvm::StringRef id, sexpr::SOp expr,
                        mlir::OpBuilder &builder,
                        std::map<llvm::StringRef, mlir::Value> &scope) {
    auto exprs = expr.getExprs();

    std::vector<mlir::Value> operands;
    for (auto e : exprs.drop_front(1)) {
      if (auto v = visitOp(e.getDefiningOp(), builder, scope)) {
        operands.emplace_back(v);
      } else {
        return {};
      }
    }

    auto var_type = sir::VarType::get(&getContext());
    return builder.create<sir::PrimOp>(
        expr->getLoc(), var_type,
        mlir::SymbolRefAttr::get(builder.getContext(), id), operands);
  }

  mlir::Value visitLambda(sexpr::SOp expr, mlir::OpBuilder &builder,
                          std::map<llvm::StringRef, mlir::Value> &scope) {
    auto exprs = expr.getExprs();
    if (exprs.size() < 2) {
      mlir::emitError(expr.getLoc())
          << "expected more than 1 argument in (lambda <args> <body>)";
      return {};
    }

    auto args_op = exprs[1].getDefiningOp();
    if (!llvm::isa<sexpr::SOp>(args_op)) {
      mlir::emitError(args_op->getLoc())
          << "expected S-expression for arguments in (lambda <args> <body>)";
      return {};
    }

    auto args = llvm::cast<sexpr::SOp>(args_op).getExprs();
    std::vector<llvm::StringRef> arg_names;
    for (auto arg : args) {
      if (auto id = llvm::dyn_cast<sexpr::IdOp>(arg.getDefiningOp())) {
        arg_names.push_back(id.getId().strref());
      } else {
        mlir::emitError(arg.getLoc()) << "expected identifier for each "
                                         "argument in (lambda <args> <body>)";
        return {};
      }
    }

    auto var_type = sir::VarType::get(&getContext());
    auto lambda = builder.create<sir::ClosureOp>(expr->getLoc(), var_type,
                                                 mlir::ValueRange{});

    auto ip = builder.saveInsertionPoint();

    auto block = &lambda.getBody().emplaceBlock();
    builder.setInsertionPointToStart(block);

    auto new_scope = scope;
    for (size_t i = 0; i < args.size(); ++i) {
      auto arg = block->addArgument(var_type, args[i].getLoc());
      new_scope.insert_or_assign(arg_names[i], arg);
    }

    mlir::Value ret;
    for (auto e : exprs.drop_front(2)) {
      if (auto v = visitOp(e.getDefiningOp(), builder, new_scope)) {
        ret = v;
      } else {
        return {};
      }
    }

    builder.create<sir::YieldOp>(exprs.back().getLoc(), ret);
    builder.restoreInsertionPoint(ip);

    return lambda;
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSExprToSIRConversionPass() {
  return std::make_unique<chira::SExprToSIRConversionPass>();
}

} // namespace chira
