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

#include "chira/conversion/sirtoscf/SIRToSCF.h"
#include "chira/dialect/chir/CHIROps.h"
#include "chira/dialect/sir/SIROps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace chira {

namespace {

struct ConvertIfOp : public mlir::OpRewritePattern<sir::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(sir::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto var_type = sir::VarType::get(getContext());
    auto cond = rewriter.create<chir::AsBoolOp>(
        op->getLoc(), rewriter.getI1Type(), op.getCond());
    auto scf_if =
        rewriter.create<mlir::scf::IfOp>(op->getLoc(), var_type, cond);
    scf_if.getThenRegion().takeBody(op.getThen());
    scf_if.getElseRegion().takeBody(op.getElse());

    rewriter.replaceOp(op, scf_if);
    return mlir::success();
  }
};

struct ConvertYieldOp : public mlir::OpRewritePattern<sir::YieldOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(sir::YieldOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<sir::IfOp>(op->getParentOp()) &&
        !llvm::isa<mlir::scf::IfOp>(op->getParentOp())) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, op.getVar());
    return mlir::success();
  }
};

struct SIRToSCFConversionPass
    : public mlir::PassWrapper<SIRToSCFConversionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, chir::CHIRDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp module = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<ConvertIfOp, ConvertYieldOp>(context);

    if (failed(mlir::applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSIRToSCFConversionPass() {
  return std::make_unique<SIRToSCFConversionPass>();
}

} // namespace chira
