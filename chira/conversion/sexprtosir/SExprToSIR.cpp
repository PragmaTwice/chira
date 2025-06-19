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
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <charconv>

namespace chira {

namespace {

struct LexScope {
  enum Kind { Global, Local, Closure };

  std::map<llvm::StringRef, mlir::Value> current_scope;
  LexScope *parent_scope = nullptr;
  Kind scope_kind = Global;
  mlir::OpBuilder &builder;
  mlir::Block *closure_block = nullptr;

  LexScope(LexScope &parent, Kind kind, mlir::OpBuilder &builder,
           mlir::Block *closure_block = nullptr)
      : parent_scope(&parent), scope_kind(kind), builder(builder),
        closure_block(closure_block) {
    if (scope_kind == Closure) {
      assert(closure_block &&
             "closure block must be provided for closure scope");
    }
  }
  LexScope(mlir::OpBuilder &builder) : builder(builder) {}
  LexScope(const LexScope &) = delete;

  mlir::Value get(llvm::StringRef id) {
    auto *scope = this;
    std::vector<LexScope *> closures;
    std::optional<mlir::Value> target_opt;
    while (scope) {
      auto it = scope->current_scope.find(id);
      if (it != scope->current_scope.end()) {
        target_opt = it->second;
        break;
      }
      if (scope->scope_kind == Closure) {
        closures.push_back(scope);
      }
      scope = scope->parent_scope;
    }

    if (!target_opt) {
      return {};
    }

    mlir::Value target = *target_opt;

    // if there is no closure barrier between the current location and the
    // target
    if (closures.empty()) {
      return target;
    }

    auto var_type = sir::VarType::get(builder.getContext());

    // capture closure variables
    for (auto it = closures.rbegin(); it != closures.rend(); ++it) {
      auto *closure_scope = *it;
      auto closure_op = closure_scope->closure_block->getParentOp();
      if (!target) {
        auto ip = builder.saveInsertionPoint();
        builder.setInsertionPoint(closure_op);
        target = builder.create<sir::UnresolvedNameOp>(
            closure_op->getLoc(), var_type,
            mlir::StringAttr::get(builder.getContext(), id));

        builder.restoreInsertionPoint(ip);
      }
      if (auto closure = llvm::dyn_cast<sir::ClosureOp>(closure_op)) {
        closure.getCapsMutable().append(target);
      } else {
        llvm_unreachable("closure op type not supported");
      }

      target =
          closure_scope->closure_block->addArgument(var_type, target.getLoc());
      closure_scope->current_scope[id] = target;
    }

    return target;
  }

  void set(llvm::StringRef id, mlir::Value value) { current_scope[id] = value; }
};

struct SExprToSIRConversionPass
    : public mlir::PassWrapper<SExprToSIRConversionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<sir::SIRDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    sexpr::RootOp root;
    module->walk([&](sexpr::RootOp op) { root = op; });

    auto new_module = mlir::ModuleOp::create(module->getLoc());
    mlir::OpBuilder builder(new_module.getRegion());

    bool success = true;
    LexScope global_scope(builder);
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
                      LexScope &scope) {
    if (auto s = llvm::dyn_cast<sexpr::SOp>(op)) {
      return visitExpr(s, builder, scope);
    } else if (auto id = llvm::dyn_cast<sexpr::IdOp>(op)) {
      if (auto v = scope.get(id.getId().strref())) {
        return v;
      } else {
        mlir::emitError(id.getLoc()) << "undefined identifier " << id.getId();
        return {};
      }
    } else if (auto str = llvm::dyn_cast<sexpr::StrOp>(op)) {
      auto var_type = sir::VarType::get(&getContext());
      return builder.create<sir::StrOp>(op->getLoc(), var_type, str.getStr());
    } else if (auto num = llvm::dyn_cast<sexpr::NumOp>(op)) {
      auto var_type = sir::VarType::get(&getContext());

      auto num_str = num.getNum().getValue();
      int64_t int_value;
      auto int_res = std::from_chars(
          num_str.data(), num_str.data() + num_str.size(), int_value);
      mlir::Attribute val_attr;
      if (int_res.ec != std::errc() ||
          int_res.ptr != num_str.data() + num_str.size()) {
        double double_value;
        std::from_chars(num_str.data(), num_str.data() + num_str.size(),
                        double_value);
        val_attr = mlir::FloatAttr::get(mlir::Float64Type::get(&getContext()),
                                        double_value);
      } else {
        val_attr = mlir::IntegerAttr::get(
            mlir::IntegerType::get(&getContext(), 64), int_value);
      }
      return builder.create<sir::NumOp>(op->getLoc(), var_type, val_attr);
    }

    llvm_unreachable("unexpected operation type");
  }

  bool isArithPrimOp(llvm::StringRef id) {
    return id == "+" || id == "-" || id == "*" || id == "/" || id == "=" ||
           id == "<" || id == "<=" || id == ">" || id == ">=";
  }

  bool isIOPrimOp(llvm::StringRef id) {
    return id == "display" || id == "newline";
  }

  bool isPrimOp(llvm::StringRef id) {
    return isArithPrimOp(id) || isIOPrimOp(id);
  }

  mlir::Value visitExpr(sexpr::SOp expr, mlir::OpBuilder &builder,
                        LexScope &scope) {
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
      } else if (opcode.getId() == "set!") {
        return visitSet(expr, builder, scope);
      } else if (opcode.getId() == "let") {
        return visitLet(expr, builder, scope);
      }
    }

    return visitCall(expr, builder, scope);
  }

  mlir::Value visitSet(sexpr::SOp expr, mlir::OpBuilder &builder,
                       LexScope &scope) {
    auto exprs = expr.getExprs();
    if (exprs.size() != 3) {
      mlir::emitError(expr.getLoc())
          << "expected 2 arguments in (set! <name> <value>)";
      return {};
    }

    auto name_op = exprs[1].getDefiningOp();
    if (auto name = llvm::dyn_cast<sexpr::IdOp>(name_op)) {
      auto var = visitOp(name, builder, scope);
      if (!var) {
        return {};
      }

      auto value = visitOp(exprs[2].getDefiningOp(), builder, scope);
      if (!value) {
        return {};
      }

      builder.create<sir::SetOp>(expr->getLoc(), var, value);
      auto unspec = builder.create<sir::UnspecifiedOp>(
          expr->getLoc(), sir::VarType::get(&getContext()));
      return unspec;
    } else {
      mlir::emitError(name_op->getLoc())
          << "expected identifier for the first operand of (set!)";
      return {};
    }
  }

  mlir::Value visitLet(sexpr::SOp expr, mlir::OpBuilder &builder,
                       LexScope &scope) {
    auto exprs = expr.getExprs();
    if (exprs.size() < 3) {
      mlir::emitError(expr.getLoc())
          << "expected at least 2 arguments in (let <bindings> <body>..)";
      return {};
    }

    auto bindings = llvm::dyn_cast<sexpr::SOp>(exprs[1].getDefiningOp());
    if (!bindings) {
      mlir::emitError(exprs[1].getLoc())
          << "expected S-expression for the bindings in (let <bindings> "
             "<body>)";
      return {};
    }

    if (bindings.getExprs().empty()) {
      mlir::emitError(exprs[1].getLoc())
          << "expected non-empty S-expression for the bindings in (let "
             "<bindings> <body>)";
      return {};
    }

    LexScope new_scope(scope, LexScope::Local, builder);
    for (auto binding : bindings.getExprs()) {
      auto binding_op = llvm::dyn_cast<sexpr::SOp>(binding.getDefiningOp());
      if (!binding_op || binding_op.getExprs().size() != 2) {
        mlir::emitError(binding.getLoc())
            << "expected (<name> <value>) for each binding in (let <bindings> "
               "<body>)";
        return {};
      }

      auto name_op =
          llvm::dyn_cast<sexpr::IdOp>(binding_op.getExprs()[0].getDefiningOp());
      if (!name_op) {
        mlir::emitError(binding_op.getExprs()[0].getLoc())
            << "expected identifier for the name in (let ((<name> <value>)..) "
               "<body>)";
        return {};
      }

      auto value =
          visitOp(binding_op.getExprs()[1].getDefiningOp(), builder, new_scope);
      if (!value) {
        return {};
      }

      value.getDefiningOp()->setAttr("defined_name", name_op.getIdAttr());
      new_scope.set(name_op.getId().strref(), value);
    }

    mlir::Value last;
    for (auto body_expr : exprs.drop_front(2)) {
      last = visitOp(body_expr.getDefiningOp(), builder, new_scope);
      if (!last) {
        return {};
      }
    }

    return last;
  }

  mlir::Value visitBegin(sexpr::SOp expr, mlir::OpBuilder &builder,
                         LexScope &scope) {
    auto exprs = expr.getExprs();
    if (exprs.size() < 2) {
      mlir::emitError(expr.getLoc())
          << "expected more than 1 argument in (begin <expr>..)";
      return {};
    }

    LexScope new_scope(scope, LexScope::Local, builder);

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
                      LexScope &scope) {
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
                        LexScope &scope) {
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
                          LexScope &scope) {
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

      // put it into the scope in ahead of time to let recursive function works
      scope.set(name.getId().strref(), mlir::Value());

      auto value = visitOp(exprs[2].getDefiningOp(), builder, scope);
      if (!value) {
        return {};
      }

      value.getDefiningOp()->setAttr("defined_name", name.getIdAttr());
      scope.set(name.getId().strref(), value);
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
      scope.set(name, lambda);

      auto ip = builder.saveInsertionPoint();

      auto block = &lambda.getBody().emplaceBlock();
      builder.setInsertionPointToStart(block);

      LexScope new_scope(scope, LexScope::Closure, builder, block);
      for (size_t i = 0; i < args.size(); ++i) {
        auto arg =
            block->addArgument(var_type, names.getExprs()[i + 1].getLoc());
        new_scope.set(args[i], arg);
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
                        mlir::OpBuilder &builder, LexScope &scope) {
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
    auto symbol = mlir::StringAttr::get(builder.getContext(), id);
    if (isArithPrimOp(id)) {
      return builder.create<sir::ArithPrimOp>(expr->getLoc(), var_type, symbol,
                                              operands);
    } else if (isIOPrimOp(id)) {
      return builder.create<sir::IOPrimOp>(expr->getLoc(), var_type, symbol,
                                           operands);
    }

    llvm_unreachable("unexpected primitive operation");
  }

  mlir::Value visitLambda(sexpr::SOp expr, mlir::OpBuilder &builder,
                          LexScope &scope) {
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

    LexScope new_scope(scope, LexScope::Closure, builder, block);
    for (size_t i = 0; i < args.size(); ++i) {
      auto arg = block->addArgument(var_type, args[i].getLoc());
      new_scope.set(arg_names[i], arg);
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
