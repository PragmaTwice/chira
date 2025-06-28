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

#include "chira/dialect/sir/SIROps.h"
#include "chira/dialect/sir/transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace chira::sir {

namespace {

struct ConvertClosure : mlir::OpRewritePattern<sir::ClosureOp> {
  ConvertClosure(mlir::MLIRContext *context, std::atomic<size_t> &lambda_count)
      : mlir::OpRewritePattern<sir::ClosureOp>(context),
        lambda_count(lambda_count) {}

  mlir::LogicalResult
  matchAndRewrite(sir::ClosureOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op->getParentOfType<sir::FuncOp>());

    auto name = "lambda_" + std::to_string(lambda_count.fetch_add(1));
    auto name_attr = mlir::StringAttr::get(getContext(), name);
    auto symbol = mlir::SymbolRefAttr::get(getContext(), name);
    auto i64 = mlir::IntegerType::get(getContext(), 64);
    auto func = rewriter.create<sir::FuncOp>(
        op->getLoc(), name_attr,
        mlir::IntegerAttr::get(i64, op.getCaps().size()));
    func.getBody().takeBody(op.getBody());

    rewriter.restoreInsertionPoint(ip);
    auto lambda_type = sir::LambdaType::get(
        getContext(), func.getBody().getNumArguments() - op.getCaps().size(),
        op.getCaps().size());
    auto func_ref =
        rewriter.create<sir::FuncRefOp>(op->getLoc(), lambda_type, symbol);
    auto var_type = sir::VarType::get(getContext());
    auto bind = rewriter.replaceOpWithNewOp<sir::BindOp>(
        op, var_type, mlir::ValueRange{func_ref}, op.getCaps());
    if (auto dn = op->getAttr("defined_name")) {
      bind->setAttr("defined_name", dn);
    }
    return mlir::success();
  }

private:
  std::atomic<size_t> &lambda_count;
};

struct ConvertLambda : mlir::OpRewritePattern<sir::LambdaOp> {
  ConvertLambda(mlir::MLIRContext *context, std::atomic<size_t> &lambda_count)
      : mlir::OpRewritePattern<sir::LambdaOp>(context),
        lambda_count(lambda_count) {}

  mlir::LogicalResult
  matchAndRewrite(sir::LambdaOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op->getParentOfType<sir::FuncOp>());

    auto name = "lambda_" + std::to_string(lambda_count.fetch_add(1));
    auto name_attr = mlir::StringAttr::get(getContext(), name);
    auto symbol = mlir::SymbolRefAttr::get(getContext(), name);
    auto func = rewriter.create<sir::FuncOp>(
        op->getLoc(), name_attr,
        rewriter.getI64IntegerAttr(op.getType().getCapSize()));
    func.getBody().takeBody(op.getBody());

    rewriter.restoreInsertionPoint(ip);
    rewriter.replaceOpWithNewOp<sir::FuncRefOp>(op, op.getType(), symbol);
    return mlir::success();
  }

private:
  std::atomic<size_t> &lambda_count;
};

struct LambdaOutliningPass
    : public mlir::PassWrapper<LambdaOutliningPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp module = getOperation();

    auto new_module = mlir::ModuleOp::create(module.getLoc());
    mlir::OpBuilder builder(context);
    builder.setInsertionPointToStart(new_module.getBody());
    auto i64 = mlir::IntegerType::get(context, 64);
    auto main = builder.create<sir::FuncOp>(
        module->getLoc(), mlir::StringAttr::get(context, "main"),
        mlir::IntegerAttr::get(i64, 0));
    main.getBody().takeBody(module.getRegion());
    builder.setInsertionPointToEnd(&main.getBody().front());
    auto var_type = sir::VarType::get(context);
    auto unspec =
        builder.create<sir::UnspecifiedOp>(module->getLoc(), var_type);
    builder.create<sir::YieldOp>(module->getLoc(), unspec);
    module.getRegion().takeBody(new_module.getRegion());

    mlir::RewritePatternSet patterns(context);
    std::atomic<size_t> lambda_count = 1;
    patterns.add<ConvertClosure, ConvertLambda>(context, lambda_count);

    if (failed(mlir::applyPatternsGreedily(
            main, std::move(patterns),
            mlir::GreedyRewriteConfig{.useTopDownTraversal = true})))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLambdaOutliningPass() {
  return std::make_unique<LambdaOutliningPass>();
}

} // namespace chira::sir
