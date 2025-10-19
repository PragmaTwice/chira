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

#include "chira/conversion/chirtofunc/CHIRToFunc.h"
#include "chira/conversion/chirtollvm/CHIRToLLVM.h"
#include "chira/conversion/sexprtosir/SExprToSIR.h"
#include "chira/conversion/sirtoscf/SIRToSCF.h"
#include "chira/dialect/sexpr/SExprPrinter.h"
#include "chira/dialect/sexpr/transforms/Passes.h"
#include "chira/dialect/sir/transforms/Passes.h"
#include "chira/parser/Parser.h"
#include "chira/runtime/Runtime.h"
#include "chira/runtime/chirart.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include <memory>
#include <system_error>

llvm::cl::OptionCategory CLICat("chira-cli options");

llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input file>"),
                                         llvm::cl::Required,
                                         llvm::cl::cat(CLICat));

llvm::cl::opt<std::string>
    OutputFilename("o", llvm::cl::desc("path for the output file"),
                   llvm::cl::init("a.out"), llvm::cl::cat(CLICat));

llvm::cl::opt<bool>
    MacroExpanderOnly("expand-macro",
                      llvm::cl::desc("only run the macro expander, and output "
                                     "the source code after expansion"),
                      llvm::cl::init(false), llvm::cl::cat(CLICat));

llvm::cl::opt<size_t>
    MaxLineLength("max-line-length",
                  llvm::cl::desc("maximum line length for the output source "
                                 "code (combined with -expand-macro, default: "
                                 "80)"),
                  llvm::cl::init(chira::sexpr::Printer::MAX_LINE_LENGTH),
                  llvm::cl::cat(CLICat));

llvm::cl::opt<bool>
    OutputSIR("sir",
              llvm::cl::desc("transform the input program to SIR and output it "
                             "in the MLIR SIR text form"),
              llvm::cl::init(false), llvm::cl::cat(CLICat));

llvm::cl::opt<bool> OutputCHIR(
    "chir",
    llvm::cl::desc("transform the input program to CHIR and output it "
                   "in the MLIR CHIR text form"),
    llvm::cl::init(false), llvm::cl::cat(CLICat));

llvm::cl::opt<bool> OutputLLVM(
    "llvmir",
    llvm::cl::desc(
        "transform the input program to LLVM IR and output it in text form"),
    llvm::cl::init(false), llvm::cl::cat(CLICat));

llvm::cl::opt<bool> OutputObject(
    "object",
    llvm::cl::desc("compile the input program to object file and dump it"),
    llvm::cl::init(false), llvm::cl::cat(CLICat));

llvm::cl::opt<bool> PrintIR(
    "print-ir",
    llvm::cl::desc("print IR before and after all transformation passes"),
    llvm::cl::init(false), llvm::cl::cat(CLICat));

int main(int argc, char *argv[]) {
  llvm::cl::HideUnrelatedOptions(CLICat);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Chira - an MLIR-based Scheme JIT compiler and runtime\n");

  mlir::MLIRContext context;
  context.disableMultithreading(PrintIR);

  auto input_res = llvm::MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (!input_res) {
    llvm::errs() << "error: failed to read file `" << InputFilename
                 << "` (due to: " << input_res.getError().message() << ")"
                 << "\n";
    return 1;
  }
  auto input = std::move(*input_res);

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(input->getMemBufferRef()),
      llvm::SMLoc());
  mlir::SourceMgrDiagnosticHandler diag_handler(source_mgr, &context);

  std::error_code ec;
  llvm::raw_fd_ostream os(OutputFilename, ec);
  if (ec) {
    llvm::errs() << "error: "
                 << "failed to open output path `" << OutputFilename
                 << "` (due to: " << ec.message() << ")"
                 << "\n";
    return 1;
  }

  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();

  chira::parser::Tokenizer tokenizer(context, input.get());

  auto res = tokenizer.Tokenize();
  if (res.failed()) {
    return 1;
  }

  auto tokens = tokenizer.GetResult();
  chira::parser::Parser parser(context, tokens);

  res = parser.Parse();
  if (res.failed()) {
    return 1;
  }

  auto module = parser.Module();

  {
    mlir::PassManager pm(module->getContext());
    if (PrintIR)
      pm.enableIRPrinting();

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(chira::sexpr::createDefineSyntaxConstructorPass());
    pm.addPass(chira::sexpr::createMacroExpanderPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    if (mlir::failed(pm.run(module))) {
      return 1;
    }
  }

  if (MacroExpanderOnly) {
    os << chira::sexpr::Printer::Print(module, 0, MaxLineLength);
    return 0;
  }

  {
    mlir::PassManager pm(module->getContext());
    if (PrintIR)
      pm.enableIRPrinting();

    pm.addPass(chira::createSExprToSIRConversionPass());
    pm.addPass(chira::sir::createRecursiveResolverPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    if (mlir::failed(pm.run(module))) {
      return 1;
    }
  }

  if (OutputSIR) {
    module->print(os, flags);
    return 0;
  }

  {
    mlir::PassManager pm(module->getContext());
    if (PrintIR)
      pm.enableIRPrinting();

    pm.addPass(chira::sir::createLambdaOutliningPass());
    pm.addPass(chira::sir::createBindLoweringPass());
    pm.addPass(chira::createSIRToSCFConversionPass());

    if (mlir::failed(pm.run(module))) {
      return 1;
    }
  }

  if (OutputCHIR) {
    module->print(os, flags);
    return 0;
  }

  {
    mlir::PassManager pm(module->getContext());
    if (PrintIR)
      pm.enableIRPrinting();

    pm.addPass(chira::createCHIRToFuncConversionPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(chira::createCHIRToLLVMConversionPass());
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());

    if (mlir::failed(pm.run(module))) {
      return 1;
    }
  }

  if (OutputLLVM) {
    module->print(os, flags);
    return 0;
  }

  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerBuiltinDialectTranslation(*module.getContext());

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  mlir::ExecutionEngineOptions options = {.llvmModuleBuilder =
                                              chira::rt::buildLLVMModule,
                                          .enableObjectDump = true};
  auto engine_exp = mlir::ExecutionEngine::create(module, options);
  if (!engine_exp) {
    llvm::errs() << "failed to initialize the execution engine: "
                 << engine_exp.takeError() << "\n";
    return 1;
  }

  auto engine = std::move(*engine_exp);
  auto main_res = engine->lookup("chiracg_main");
  if (!main_res) {
    llvm::errs() << "failed to lookup entry function: " << main_res.takeError()
                 << "\n";
    return 1;
  }

  if (OutputObject) {
    engine->dumpToObjectFile(OutputFilename);
    return 0;
  }

  auto main = reinterpret_cast<chirart::Lambda>(*main_res);

  chirart::Var var;
  main(&var, nullptr, nullptr);
}
