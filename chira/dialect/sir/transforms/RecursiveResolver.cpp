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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace chira::sir {

namespace {

struct ConvertSelfRecursive : mlir::OpRewritePattern<sir::ClosureOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(sir::ClosureOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto caps = op.getCaps();
    auto it = std::find(caps.begin(), caps.end(), op.getRes());
    if (it == caps.end()) {
      return rewriter.notifyMatchFailure(op, "not a self-recursive closure");
    }

    auto lambda_type = sir::LambdaType::get(getContext());
    auto i64 = mlir::IntegerType::get(getContext(), 64);
    auto lambda = rewriter.create<sir::LambdaOp>(
        op->getLoc(), lambda_type, mlir::IntegerAttr::get(i64, caps.size()));
    lambda.getBody().takeBody(op.getBody());

    auto var_type = sir::VarType::get(getContext());

    std::vector<mlir::Value> new_caps;
    for (auto cap : caps) {
      new_caps.push_back(cap == op.getRes() ? lambda : cap);
    }
    rewriter.replaceOpWithNewOp<sir::BindOp>(
        op, var_type, mlir::ValueRange{lambda}, new_caps);
    return mlir::success();
  }
};

struct RecursiveResolverPass
    : public mlir::PassWrapper<RecursiveResolverPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp module = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<ConvertSelfRecursive>(context);

    if (failed(mlir::applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createRecursiveResolverPass() {
  return std::make_unique<RecursiveResolverPass>();
}

} // namespace chira::sir
