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

struct ConvertStrOp : mlir::ConvertOpToLLVMPattern<sir::StrOp> {
  using ConvertOpToLLVMPattern<sir::StrOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::StrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto str = op.getStr().getValue();
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(
        op->getParentOfType<mlir::ModuleOp>().getBody());
    auto name = "chiracg_str_" + llvm::utohexstr(llvm::hash_value(str), true);
    auto type =
        mlir::LLVM::LLVMArrayType::get(rewriter.getI8Type(), str.size() + 1);
    auto str_with_null = str.str() + '\0';
    auto global = rewriter.create<mlir::LLVM::GlobalOp>(
        op->getLoc(), type, true, mlir::LLVM::Linkage::Internal, name,
        rewriter.getStringAttr(str_with_null));
    rewriter.restoreInsertionPoint(ip);

    auto var = allocVar(op, rewriter);
    auto ptr = rewriter.create<mlir::LLVM::AddressOfOp>(op->getLoc(), global);
    makeLLVMFuncCall("chirart_string", op, rewriter, getVoidType(),
                     {var, ptr,
                      rewriter.create<mlir::LLVM::ConstantOp>(
                          op->getLoc(), rewriter.getI64Type(),
                          rewriter.getI64IntegerAttr(str.size()))});
    rewriter.replaceOp(op, var);
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

template <typename Op> struct ConvertPrimOp : mlir::ConvertOpToLLVMPattern<Op> {
  using mlir::ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(Op op,
                  typename mlir::ConvertOpToLLVMPattern<Op>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto opcode = adaptor.getOp().getValue();
    auto address = makeLLVMFuncAddr(
        "chirart_" + opcode.str(), op, rewriter, this->getVoidType(),
        {this->getVoidPtrType(), this->getVoidPtrType(),
         this->getVoidPtrType()});

    auto var = allocVar(op, rewriter);
    auto op_info = sir::PrimOps::getByName(opcode).value();
    auto arity = rewriter.create<mlir::LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI64Type(), op_info.num_args.Encoded());
    makeLLVMFuncCall("chirart_prim_op", op, rewriter, this->getVoidType(),
                     {var, address, arity});
    rewriter.replaceOp(op, var);
    return mlir::success();
  }
};

struct ConvertArithPrimOp : ConvertPrimOp<sir::ArithPrimOp> {
  using ConvertPrimOp<sir::ArithPrimOp>::ConvertPrimOp;
};
struct ConvertIOPrimOp : ConvertPrimOp<sir::IOPrimOp> {
  using ConvertPrimOp<sir::IOPrimOp>::ConvertPrimOp;
};

struct ConvertCallOp : mlir::ConvertOpToLLVMPattern<sir::CallOp> {
  using ConvertOpToLLVMPattern<sir::CallOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(sir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(getContext());

    auto var = allocVar(op, rewriter);
    auto one = rewriter.create<mlir::LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
    auto array_type = mlir::LLVM::LLVMArrayType::get(
        rewriter.getI64Type(), 1 + 3 * adaptor.getArgs().size());
    auto args = rewriter.create<mlir::LLVM::AllocaOp>(op->getLoc(), ptr_type,
                                                      array_type, one);
    makeLLVMFuncCall(
        "chirart_args_set_size", op, rewriter, getVoidType(),
        {args, rewriter.create<mlir::LLVM::ConstantOp>(
                   op->getLoc(), rewriter.getI64Type(),
                   rewriter.getI64IntegerAttr(adaptor.getArgs().size()))});
    size_t i = 0;
    for (auto arg : adaptor.getArgs()) {
      makeLLVMFuncCall("chirart_args_store", op, rewriter, getVoidType(),
                       {args,
                        rewriter.create<mlir::LLVM::ConstantOp>(
                            op->getLoc(), rewriter.getI64Type(),
                            rewriter.getI64IntegerAttr(i++)),
                        arg});
    }
    makeLLVMFuncCall("chirart_call", op, rewriter, getVoidType(),
                     {var, adaptor.getCallee(), args});
    rewriter.replaceOp(op, var);
    return mlir::success();
  }
};

} // namespace

void populateSIRToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                         mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertNumOp, ConvertUnspecOp, ConvertArithPrimOp,
               ConvertIOPrimOp, ConvertCallOp, ConvertSetOp, ConvertStrOp>(
      converter);
}

} // namespace chira
