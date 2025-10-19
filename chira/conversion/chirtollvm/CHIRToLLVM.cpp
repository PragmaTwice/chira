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

#include "chira/conversion/chirtollvm/CHIRToLLVM.h"
#include "chira/conversion/sirtollvm/SIRToLLVM.h"
#include "chira/dialect/chir/CHIROps.h"
#include "chira/dialect/sir/PrimOps.h"
#include "chira/dialect/sir/SIROps.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace chira {

namespace {

mlir::LLVM::AddressOfOp
makeLLVMFuncAddr(llvm::StringRef func_name, mlir::Operation *op,
                 mlir::ConversionPatternRewriter &rewriter,
                 mlir::Type return_type, llvm::ArrayRef<mlir::Type> arg_types) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto func_type = mlir::LLVM::LLVMFunctionType::get(return_type, arg_types);
  if (!module.lookupSymbol(func_name)) {
    auto ip = rewriter.saveInsertionPoint();
    auto block = &module.getRegion().front();
    rewriter.setInsertionPointToStart(block);

    rewriter.create<mlir::LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name,
                                            func_type);

    rewriter.restoreInsertionPoint(ip);
  }

  auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  return rewriter.create<mlir::LLVM::AddressOfOp>(op->getLoc(), ptr_type,
                                                  func_name);
}

mlir::LLVM::CallOp makeLLVMFuncCall(llvm::StringRef func_name,
                                    mlir::Operation *op,
                                    mlir::ConversionPatternRewriter &rewriter,
                                    mlir::Type return_type,
                                    mlir::ValueRange args) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto func_type = mlir::LLVM::LLVMFunctionType::get(
      return_type,
      llvm::map_to_vector(args, [](mlir::Value v) { return v.getType(); }));
  if (!module.lookupSymbol(func_name)) {
    auto ip = rewriter.saveInsertionPoint();
    auto block = &module.getRegion().front();
    rewriter.setInsertionPointToStart(block);

    rewriter.create<mlir::LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name,
                                            func_type);

    rewriter.restoreInsertionPoint(ip);
  }

  return rewriter.create<mlir::LLVM::CallOp>(op->getLoc(), func_type, func_name,
                                             args);
}

mlir::LLVM::AllocaOp allocVar(mlir::Operation *op,
                              mlir::ConversionPatternRewriter &rewriter) {
  auto ptr_type = mlir::LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  auto one = rewriter.create<mlir::LLVM::ConstantOp>(
      op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
  auto array_type = mlir::LLVM::LLVMArrayType::get(rewriter.getI64Type(), 3);
  return rewriter.create<mlir::LLVM::AllocaOp>(op->getLoc(), ptr_type,
                                               array_type, one);
}

struct ConvertEnvLoadOp : mlir::ConvertOpToLLVMPattern<chir::EnvLoadOp> {
  using ConvertOpToLLVMPattern<chir::EnvLoadOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(chir::EnvLoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(getContext());

    auto idx = rewriter.create<mlir::LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI64Type(), adaptor.getIndex().getInt());
    auto call = makeLLVMFuncCall("chirart_env_load", op, rewriter, ptr_type,
                                 {adaptor.getEnv(), idx});
    rewriter.replaceOp(op, call);
    return mlir::success();
  }
};

struct ConvertArgsLoadOp : mlir::ConvertOpToLLVMPattern<chir::ArgsLoadOp> {
  using ConvertOpToLLVMPattern<chir::ArgsLoadOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(chir::ArgsLoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(getContext());

    auto idx = rewriter.create<mlir::LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI64Type(), adaptor.getIndex().getInt());
    auto call = makeLLVMFuncCall("chirart_args_load", op, rewriter, ptr_type,
                                 {adaptor.getArgs(), idx});
    rewriter.replaceOp(op, call);
    return mlir::success();
  }
};

struct ConvertEnvStoreOp : mlir::ConvertOpToLLVMPattern<chir::EnvStoreOp> {
  using ConvertOpToLLVMPattern<chir::EnvStoreOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(chir::EnvStoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(getContext());

    auto idx = rewriter.create<mlir::LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI64Type(), adaptor.getIndex().getInt());
    auto call = makeLLVMFuncCall("chirart_env_store", op, rewriter, ptr_type,
                                 {adaptor.getEnv(), idx, adaptor.getVar()});
    rewriter.replaceOp(op, call);
    return mlir::success();
  }
};

struct ConvertClosureOp : mlir::ConvertOpToLLVMPattern<chir::ClosureOp> {
  using ConvertOpToLLVMPattern<chir::ClosureOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(chir::ClosureOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lambda_type =
        llvm::dyn_cast<sir::LambdaType>(op.getLambda().getType());
    if (!lambda_type) {
      llvm_unreachable("should be a lambda type");
    }
    auto cap_size = rewriter.create<mlir::LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI64Type(), lambda_type.getParamSize());
    auto var = allocVar(op, rewriter);
    makeLLVMFuncCall("chirart_closure", op, rewriter, getVoidType(),
                     {var, adaptor.getLambda(), adaptor.getEnv(), cap_size});
    rewriter.replaceOp(op, var);
    return mlir::success();
  }
};

struct ConvertAsBoolOp : mlir::ConvertOpToLLVMPattern<chir::AsBoolOp> {
  using ConvertOpToLLVMPattern<chir::AsBoolOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(chir::AsBoolOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto call = makeLLVMFuncCall("chirart_get_bool", op, rewriter,
                                 rewriter.getI1Type(), {adaptor.getVar()});
    rewriter.replaceOp(op, call);
    return mlir::success();
  }
};

struct ConvertEnvOp : mlir::ConvertOpToLLVMPattern<chir::EnvOp> {
  using ConvertOpToLLVMPattern<chir::EnvOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(chir::EnvOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(getContext());
    auto env_type = llvm::dyn_cast<chir::EnvType>(op->getResultTypes()[0]);
    if (!env_type) {
      llvm_unreachable("should be an env type");
    }
    auto array_type =
        mlir::LLVM::LLVMArrayType::get(ptr_type, env_type.getSize());
    auto one = rewriter.create<mlir::LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
    auto env = rewriter.create<mlir::LLVM::AllocaOp>(op->getLoc(), ptr_type,
                                                     array_type, one);
    rewriter.replaceOp(op, env);
    return mlir::success();
  }
};

struct ConvertFuncRefOp : mlir::ConvertOpToLLVMPattern<chir::FuncRefOp> {
  using ConvertOpToLLVMPattern<chir::FuncRefOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(chir::FuncRefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(getContext());
    auto address = rewriter.create<mlir::LLVM::AddressOfOp>(
        op->getLoc(), ptr_type,
        "chiracg_" + adaptor.getFunc().getRootReference().str());
    rewriter.replaceOp(op, address);
    return mlir::success();
  }
};

struct CHIRToLLVMConversionPass
    : public mlir::PassWrapper<CHIRToLLVMConversionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp module = getOperation();

    mlir::LLVMTypeConverter converter(context);
    mlir::RewritePatternSet patterns(context);

    converter.addConversion([&](sir::VarType) {
      return mlir::LLVM::LLVMPointerType::get(context);
    });
    converter.addConversion([&](sir::LambdaType) {
      return mlir::LLVM::LLVMPointerType::get(context);
    });
    converter.addConversion([&](chir::EnvType) {
      return mlir::LLVM::LLVMPointerType::get(context);
    });
    converter.addConversion([&](chir::ArgsType) {
      return mlir::LLVM::LLVMPointerType::get(context);
    });

    mlir::ConversionTarget target(*context);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    populateSIRToLLVMConversionPatterns(converter, patterns);
    populateCHIRToLLVMConversionPatterns(converter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(converter, patterns);

    if (failed(
            mlir::applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void populateCHIRToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                          mlir::RewritePatternSet &patterns) {
  patterns
      .add<ConvertEnvLoadOp, ConvertAsBoolOp, ConvertEnvOp, ConvertEnvStoreOp,
           ConvertClosureOp, ConvertFuncRefOp, ConvertArgsLoadOp>(converter);
}

std::unique_ptr<mlir::Pass> createCHIRToLLVMConversionPass() {
  return std::make_unique<CHIRToLLVMConversionPass>();
}

} // namespace chira
