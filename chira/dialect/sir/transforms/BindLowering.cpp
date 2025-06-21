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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace chira::sir {

namespace {

struct ConvertBindOp : public mlir::OpRewritePattern<sir::BindOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(sir::BindOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getLambdas().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "not implemented yet for multiple lambdas");
    }

    auto lambda = op.getLambdas().front();
    auto caps = op.getCaps();
    auto env_type = sir::EnvType::get(getContext(), caps.size());
    auto env = rewriter.create<EnvOp>(op->getLoc(), env_type);
    auto var_type = sir::VarType::get(getContext());
    auto closure = rewriter.create<sir::ClosureFromEnvOp>(
        op->getLoc(), var_type, lambda, env);
    size_t idx = 0;
    for (auto cap : caps) {
      rewriter.create<EnvStoreOp>(
          op->getLoc(), env, rewriter.getI64IntegerAttr(idx++),
          llvm::isa<LambdaType>(cap.getType()) ? closure : cap);
    }
    rewriter.replaceOp(op, closure);
    return mlir::success();
  }
};

struct ConvertBindRefOp : public mlir::OpRewritePattern<sir::BindRefOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(sir::BindRefOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto lambda_type = sir::LambdaType::get(getContext());
    auto ref = rewriter.create<sir::FuncRefOp>(op->getLoc(), lambda_type,
                                               op.getFunc());
    auto var_type = sir::VarType::get(getContext());
    rewriter.replaceOpWithNewOp<sir::BindOp>(
        op, var_type, mlir::ValueRange{ref}, op.getCaps());
    return mlir::success();
  }
};

struct BindLoweringPass
    : public mlir::PassWrapper<BindLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp module = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<ConvertBindOp, ConvertBindRefOp>(context);

    if (failed(mlir::applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createBindLoweringPass() {
  return std::make_unique<BindLoweringPass>();
}

} // namespace chira::sir
