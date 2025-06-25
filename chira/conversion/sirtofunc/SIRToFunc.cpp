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

#include "chira/conversion/sirtofunc/SIRToFunc.h"
#include "chira/dialect/sir/SIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cstddef>

namespace chira {

namespace {

struct ConvertFuncOp : public mlir::OpConversionPattern<sir::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(sir::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto cap_size = op.getCapSize().getUInt();
    auto param_size = op.getBody().getNumArguments() - cap_size;
    auto var_type = sir::VarType::get(getContext());
    auto env_type = sir::EnvType::get(getContext(), cap_size);
    auto args_type = sir::ArgsType::get(getContext(), param_size);
    std::vector<mlir::Type> arg_types{var_type, args_type, env_type};
    auto type = rewriter.getFunctionType(arg_types, {});
    std::string name = "chiracg_" + op.getSymName().getValue().str();
    auto func = rewriter.create<mlir::func::FuncOp>(op->getLoc(), name, type);

    auto body = &op.getBody();
    mlir::OpBuilder builder(getContext());
    builder.setInsertionPointToStart(&body->front());
    mlir::Value env = body->insertArgument(0u, env_type, op.getLoc());
    mlir::Value args = body->insertArgument(0u, args_type, op.getLoc());
    body->insertArgument(0u, var_type, op.getLoc());
    size_t param_begin = 3;
    for (size_t i = param_begin; i < param_begin + param_size; ++i) {
      auto arg = body->getArgument(i);

      auto env_arg = builder.create<sir::ArgsLoadOp>(
          arg.getLoc(), var_type, args,
          builder.getI64IntegerAttr(i - param_begin));
      arg.replaceAllUsesWith(env_arg);
    }
    size_t cap_begin = param_begin + param_size;
    for (size_t i = cap_begin; i < cap_begin + cap_size; ++i) {
      auto arg = body->getArgument(i);

      auto env_arg = builder.create<sir::EnvLoadOp>(
          arg.getLoc(), var_type, env,
          builder.getI64IntegerAttr(i - cap_begin));
      arg.replaceAllUsesWith(env_arg);
    }
    body->front().eraseArguments(param_begin, param_size + cap_size);

    func.getBody().takeBody(op.getBody());
    rewriter.replaceOp(op, func);
    return mlir::success();
  }
};

struct ConvertYieldOp : public mlir::OpConversionPattern<sir::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(sir::YieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!llvm::isa<sir::FuncOp>(op->getParentOp()) &&
        !llvm::isa<mlir::func::FuncOp>(op->getParentOp())) {
      return mlir::failure();
    }

    rewriter.create<sir::SetOp>(
        op->getLoc(), op->getParentRegion()->getArgument(0), op.getVar());
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op);
    return mlir::success();
  }
};

struct SIRToFuncConversionPass
    : public mlir::PassWrapper<SIRToFuncConversionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp module = getOperation();

    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalOp<sir::SetOp>();

    patterns.add<ConvertYieldOp, ConvertFuncOp>(context);

    if (failed(
            mlir::applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSIRToFuncConversionPass() {
  return std::make_unique<SIRToFuncConversionPass>();
}

} // namespace chira
