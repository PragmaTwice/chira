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

#include "chira/conversion/sirtollvm/SIRToLLVM.h"
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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace chira {

namespace {

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

struct ConvertNumOp : mlir::ConvertOpToLLVMPattern<sir::NumOp> {
  using ConvertOpToLLVMPattern<sir::NumOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::NumOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto num = op.getNum();
    auto var = allocVar(op, rewriter);
    if (auto i = llvm::dyn_cast<mlir::IntegerAttr>(num)) {
      auto constant = rewriter.create<mlir::LLVM::ConstantOp>(
          op->getLoc(), rewriter.getI64Type(), i);

      makeLLVMFuncCall("chirart_int", op, rewriter, getVoidType(),
                       {var, constant});
      rewriter.replaceOp(op, var);
      return mlir::success();
    } else if (auto f = llvm::dyn_cast<mlir::FloatAttr>(num)) {
      auto constant = rewriter.create<mlir::LLVM::ConstantOp>(
          op->getLoc(), rewriter.getF64Type(), f);

      makeLLVMFuncCall("chirart_float", op, rewriter, getVoidType(),
                       {var, constant});
      rewriter.replaceOp(op, var);
      return mlir::success();
    }

    llvm_unreachable("attribute type not supported");
  }
};

struct ConvertUnspecOp : mlir::ConvertOpToLLVMPattern<sir::UnspecifiedOp> {
  using ConvertOpToLLVMPattern<sir::UnspecifiedOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::UnspecifiedOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto void_type = mlir::LLVM::LLVMVoidType::get(getContext());
    auto var = allocVar(op, rewriter);
    makeLLVMFuncCall("chirart_unspec", op, rewriter, void_type, {var});
    rewriter.replaceOp(op, var);
    return mlir::success();
  }
};

struct ConvertEnvLoadOp : mlir::ConvertOpToLLVMPattern<sir::EnvLoadOp> {
  using ConvertOpToLLVMPattern<sir::EnvLoadOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::EnvLoadOp op, OpAdaptor adaptor,
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

struct ConvertEnvStoreOp : mlir::ConvertOpToLLVMPattern<sir::EnvStoreOp> {
  using ConvertOpToLLVMPattern<sir::EnvStoreOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::EnvStoreOp op, OpAdaptor adaptor,
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

struct ConvertSetOp : mlir::ConvertOpToLLVMPattern<sir::SetOp> {
  using ConvertOpToLLVMPattern<sir::SetOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::SetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto call = makeLLVMFuncCall("chirart_set", op, rewriter, getVoidType(),
                                 {adaptor.getVar(), adaptor.getValue()});
    rewriter.replaceOp(op, call);
    return mlir::success();
  }
};

struct ConvertClosureFromEnvOp
    : mlir::ConvertOpToLLVMPattern<sir::ClosureFromEnvOp> {
  using ConvertOpToLLVMPattern<sir::ClosureFromEnvOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::ClosureFromEnvOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto env_type = llvm::dyn_cast<sir::EnvType>(op.getEnv().getType());
    if (!env_type) {
      llvm_unreachable("should be an env type");
    }
    auto cap_size = rewriter.create<mlir::LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI64Type(), env_type.getSize());
    auto var = allocVar(op, rewriter);
    makeLLVMFuncCall("chirart_closure", op, rewriter, getVoidType(),
                     {var, adaptor.getLambda(), adaptor.getEnv(), cap_size});
    rewriter.replaceOp(op, var);
    return mlir::success();
  }
};

struct ConvertAsBoolOp : mlir::ConvertOpToLLVMPattern<sir::AsBoolOp> {
  using ConvertOpToLLVMPattern<sir::AsBoolOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::AsBoolOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto call = makeLLVMFuncCall("chirart_get_bool", op, rewriter,
                                 rewriter.getI1Type(), {adaptor.getVar()});
    rewriter.replaceOp(op, call);
    return mlir::success();
  }
};

struct ConvertArithPrimOp : mlir::ConvertOpToLLVMPattern<sir::ArithPrimOp> {
  using ConvertOpToLLVMPattern<sir::ArithPrimOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::ArithPrimOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto opcode = adaptor.getOp().getValue();
    auto var = allocVar(op, rewriter);
    std::vector<mlir::Value> args{var};
    std::copy(adaptor.getArgs().begin(), adaptor.getArgs().end(),
              std::back_inserter(args));

    std::map<llvm::StringRef, llvm::StringRef> arith_ops = {
        {"+", "chirart_add"},      {"-", "chirart_subtract"},
        {"*", "chirart_multiply"}, {"/", "chirart_divide"},
        {"<", "chirart_lt"},       {"<=", "chirart_le"},
        {">", "chirart_gt"},       {">=", "chirart_ge"},
        {"=", "chirart_eq"},
    };

    if (auto it = arith_ops.find(opcode); it != arith_ops.end()) {
      makeLLVMFuncCall(it->second, op, rewriter, getVoidType(), args);
      rewriter.replaceOp(op, var);
      return mlir::success();
    }

    llvm_unreachable("not implemented for this arith prim op yet");
  }
};

struct ConvertIOPrimOp : mlir::ConvertOpToLLVMPattern<sir::IOPrimOp> {
  using ConvertOpToLLVMPattern<sir::IOPrimOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::IOPrimOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto opcode = adaptor.getOp().getValue();
    auto var = allocVar(op, rewriter);
    std::vector<mlir::Value> args{var};
    std::copy(adaptor.getArgs().begin(), adaptor.getArgs().end(),
              std::back_inserter(args));
    std::map<llvm::StringRef, llvm::StringRef> io_ops = {
        {"display", "chirart_display"},
        {"newline", "chirart_newline"},
    };

    if (auto it = io_ops.find(opcode); it != io_ops.end()) {
      makeLLVMFuncCall(it->second, op, rewriter, getVoidType(), args);
      rewriter.replaceOp(op, var);
      return mlir::success();
    }

    llvm_unreachable("not implemented for this IO prim op yet");
  }
};

struct ConvertCallOp : mlir::ConvertOpToLLVMPattern<sir::CallOp> {
  using ConvertOpToLLVMPattern<sir::CallOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(getContext());
    auto lambda = makeLLVMFuncCall("chirart_get_func_ptr", op, rewriter,
                                   ptr_type, {adaptor.getCallee()});
    auto env = makeLLVMFuncCall("chirart_get_caps", op, rewriter, ptr_type,
                                {adaptor.getCallee()});
    std::vector<mlir::Value> args{lambda.getResult()};
    auto var = allocVar(op, rewriter);
    args.push_back(var);
    args.push_back(env.getResult());
    std::copy(adaptor.getArgs().begin(), adaptor.getArgs().end(),
              std::back_inserter(args));
    auto func_type = mlir::LLVM::LLVMFunctionType::get(
        getVoidType(), std::vector<mlir::Type>(args.size() - 1, ptr_type));

    rewriter.create<mlir::LLVM::CallOp>(op->getLoc(), func_type, args);
    rewriter.replaceOp(op, var);
    return mlir::success();
  }
};

struct ConvertEnvOp : mlir::ConvertOpToLLVMPattern<sir::EnvOp> {
  using ConvertOpToLLVMPattern<sir::EnvOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::EnvOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(getContext());
    auto env_type = llvm::dyn_cast<sir::EnvType>(op->getResultTypes()[0]);
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

struct ConvertFuncRefOp : mlir::ConvertOpToLLVMPattern<sir::FuncRefOp> {
  using ConvertOpToLLVMPattern<sir::FuncRefOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::FuncRefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(getContext());
    auto address = rewriter.create<mlir::LLVM::AddressOfOp>(
        op->getLoc(), ptr_type,
        "chiracg_" + adaptor.getFunc().getRootReference().str());
    rewriter.replaceOp(op, address);
    return mlir::success();
  }
};

struct SIRToLLVMConversionPass
    : public mlir::PassWrapper<SIRToLLVMConversionPass,
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
    converter.addConversion([&](sir::EnvType) {
      return mlir::LLVM::LLVMPointerType::get(context);
    });

    mlir::ConversionTarget target(*context);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    patterns.add<ConvertNumOp, ConvertUnspecOp, ConvertEnvLoadOp,
                 ConvertAsBoolOp, ConvertArithPrimOp, ConvertIOPrimOp,
                 ConvertCallOp, ConvertEnvOp, ConvertEnvStoreOp,
                 ConvertClosureFromEnvOp, ConvertFuncRefOp, ConvertSetOp>(
        converter);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(converter, patterns);

    if (failed(
            mlir::applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSIRToLLVMConversionPass() {
  return std::make_unique<SIRToLLVMConversionPass>();
}

} // namespace chira
