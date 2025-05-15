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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"

namespace chira::sexpr {

namespace {

struct ConversionContext {
  bool has_error = false;

  void SetError() { has_error = true; }
  bool Good() { return !has_error; }
};

template <typename Op> struct RewritePattern : mlir::OpRewritePattern<Op> {
  RewritePattern(mlir::MLIRContext *ctx, ConversionContext &cvt_ctx)
      : mlir::OpRewritePattern<SOp>(ctx), cvt_ctx(cvt_ctx) {}

  auto emitError(mlir::Location loc) const {
    cvt_ctx.SetError();
    return mlir::emitError(loc);
  }

  ConversionContext &cvt_ctx;
};

struct ConvertSOp : RewritePattern<SOp> {
  using RewritePattern<SOp>::RewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SOp op, mlir::PatternRewriter &rewriter) const override {
    auto exprs = op.getExprs();
    if (exprs.empty())
      return mlir::failure();

    auto op_name = llvm::dyn_cast<IdOp>(exprs.front().getDefiningOp());
    if (!op_name || op_name.getId().getValue() != "define-syntax")
      return mlir::failure();

    if (exprs.size() != 3) {
      emitError(op->getLoc()) << "expected 2 operands in (define-syntax ..)";
      return mlir::failure();
    }

    auto name = llvm::dyn_cast<IdOp>(exprs[1].getDefiningOp());
    if (!name) {
      emitError(exprs[1].getLoc())
          << "expected identifier for <name> in (define-syntax <name> ..)";
      return mlir::failure();
    }

    auto syntax_rules = llvm::dyn_cast<SOp>(exprs[2].getDefiningOp());
    if (!syntax_rules) {
      emitError(exprs[2].getLoc())
          << "expected S-expr for the 2nd operand of (define-syntax ..)";
      return mlir::failure();
    }

    auto syntax_rules_exprs = syntax_rules.getExprs();
    auto sr_op_name =
        llvm::dyn_cast<IdOp>(syntax_rules_exprs.front().getDefiningOp());
    if (!sr_op_name || sr_op_name.getId() != "syntax-rules" ||
        syntax_rules_exprs.size() < 3) {
      emitError(syntax_rules->getLoc())
          << "expected (syntax-rules ..) in (define-syntax ..)";
      return mlir::failure();
    }

    auto literal_list =
        llvm::dyn_cast<SOp>(syntax_rules_exprs[1].getDefiningOp());
    if (!literal_list) {
      emitError(syntax_rules_exprs[1].getLoc())
          << "expected literal list in (syntax-rules ..)";
      return mlir::failure();
    }

    for (auto e : literal_list.getExprs()) {
      if (!llvm::isa<IdOp>(e.getDefiningOp())) {
        emitError(e.getLoc())
            << "expected all identifiers for literal list in (syntax-rules ..)";
        return mlir::failure();
      }
    }

    auto patterns = syntax_rules_exprs.drop_front(2);
    for (auto pattern : patterns) {
      auto p = llvm::dyn_cast<SOp>(pattern.getDefiningOp());
      if (!p) {
        emitError(pattern.getLoc())
            << "expected S-expr for patterns in (syntax-rules ..)";
        return mlir::failure();
      }

      auto p_exprs = p.getExprs();
      if (p_exprs.size() != 2 || !llvm::isa<SOp>(p_exprs[0].getDefiningOp())) {
        emitError(pattern.getLoc())
            << "expected (<pattern> <template>) for patterns in (syntax-rules "
               "..)";
        return mlir::failure();
      }

      auto p_lhs = llvm::dyn_cast<SOp>(p_exprs[0].getDefiningOp()).getExprs();
      if (p_lhs.empty() || !llvm::isa<IdOp>(p_lhs[0].getDefiningOp()) ||
          llvm::dyn_cast<IdOp>(p_lhs[0].getDefiningOp()).getId() !=
              name.getId()) {
        emitError(pattern.getLoc())
            << "patterns should be started by the syntax name in (syntax-rules "
               "..)";
        return mlir::failure();
      }
    }

    auto expr_type = sexpr::ExprType::get(getContext());
    rewriter.replaceOpWithNewOp<sexpr::DefineSyntaxOp>(op, expr_type, name,
                                                       literal_list, patterns);
    return mlir::success();
  }
};

struct DefineSyntaxConstructorPass
    : public mlir::PassWrapper<DefineSyntaxConstructorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp module = getOperation();

    ConversionContext cvt_ctx;
    mlir::RewritePatternSet patterns(context);
    patterns.add<ConvertSOp>(context, cvt_ctx);

    if (failed(mlir::applyPatternsGreedily(module, std::move(patterns))) ||
        !cvt_ctx.Good())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createDefineSyntaxConstructorPass() {
  return std::make_unique<DefineSyntaxConstructorPass>();
}

} // namespace chira::sexpr
