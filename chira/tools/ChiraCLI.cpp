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

#include "chira/conversion/sexprtosir/SExprToSIR.h"
#include "chira/conversion/sirtofunc/SIRToFunc.h"
#include "chira/conversion/sirtollvm/SIRToLLVM.h"
#include "chira/conversion/sirtoscf/SIRToSCF.h"
#include "chira/dialect/sexpr/SExprPrinter.h"
#include "chira/dialect/sexpr/transforms/Passes.h"
#include "chira/dialect/sir/transforms/Passes.h"
#include "chira/parser/Parser.h"
#include "chira/runtime/Runtime.h"
#include "chira/target/LLVMTarget.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
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
    MacroExpanderOnly("E",
                      llvm::cl::desc("only run the macro expander, and output "
                                     "the source code after expansion"),
                      llvm::cl::init(false), llvm::cl::cat(CLICat));

llvm::cl::opt<size_t>
    MaxLineLength("max-line-length",
                  llvm::cl::desc("maximum line length for the output source "
                                 "code (combined with -E, default: "
                                 "80)"),
                  llvm::cl::init(chira::sexpr::Printer::MAX_LINE_LENGTH),
                  llvm::cl::cat(CLICat));

llvm::cl::opt<bool>
    OutputSIR("s",
              llvm::cl::desc("transform the input program to SIR and output it "
                             "in the MLIR SIR text form"),
              llvm::cl::init(false), llvm::cl::cat(CLICat));

llvm::cl::opt<bool> OutputLLVM(
    "l",
    llvm::cl::desc(
        "transform the input program to LLVM IR and output it in text form"),
    llvm::cl::init(false), llvm::cl::cat(CLICat));

llvm::cl::opt<bool> OutputLinkedLLVM(
    "m",
    llvm::cl::desc("transform the input program to LLVM IR, link it with "
                   "chirart and output it in text form"),
    llvm::cl::init(false), llvm::cl::cat(CLICat));

llvm::cl::opt<size_t>
    OptimizationLevel("O",
                      llvm::cl::desc("optimization level (0-3, default: 3)"),
                      llvm::cl::init(3), llvm::cl::cat(CLICat));

llvm::cl::opt<bool> PrintIR(
    "p", llvm::cl::desc("print IR after and before all transformation passes"),
    llvm::cl::init(false), llvm::cl::cat(CLICat));

int main(int argc, char *argv[]) {
  llvm::cl::HideUnrelatedOptions(CLICat);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Chira CLI - MLIR-based Scheme compiler and runtime\n");

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
    pm.addPass(chira::createSIRToFuncConversionPass());
    pm.addPass(chira::createSIRToSCFConversionPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(chira::createSIRToLLVMConversionPass());
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());

    if (mlir::failed(pm.run(module))) {
      return 1;
    }
  }

  llvm::LLVMContext llvm_context;
  auto llvm_module = chira::target::translateToLLVM(
      module, llvm_context, input->getBufferIdentifier());
  if (!llvm_module) {
    return 1;
  }

  if (OutputLLVM) {
    llvm_module->print(os, nullptr);
    return 0;
  }

  if (llvm::Linker::linkModules(*llvm_module,
                                chira::rt::createRuntimeModule(llvm_context))) {
    llvm::errs() << "failed to link chirart\n";
    return 1;
  }

  if (llvm::Linker::linkModules(
          *llvm_module, chira::rt::createRuntimeLibcMainModule(llvm_context))) {
    llvm::errs() << "failed to link chirart_main\n";
    return 1;
  }

  if (OutputLinkedLLVM) {
    llvm_module->print(os, nullptr);
    return 0;
  }

  llvm::OptimizationLevel opt_levels[] = {
      llvm::OptimizationLevel::O0, llvm::OptimizationLevel::O1,
      llvm::OptimizationLevel::O2, llvm::OptimizationLevel::O3};
  if (OptimizationLevel >= std::size(opt_levels)) {
    llvm::errs() << "error: invalid optimization level: " << OptimizationLevel
                 << " (must be in range [0, 3])\n";
    return 1;
  }

  chira::target::optimizeLLVMModule(*llvm_module,
                                    opt_levels[OptimizationLevel]);

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  if (auto err = chira::target::emitObjectFile(*llvm_module, os)) {
    llvm::errs() << err << "\n";
    return 1;
  }
}
