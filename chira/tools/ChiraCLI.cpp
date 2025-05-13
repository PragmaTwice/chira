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

#include "chira/parser/Parser.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

llvm::cl::OptionCategory CLICat("chira-cli options");

llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input file>"),
                                         llvm::cl::Required,
                                         llvm::cl::cat(CLICat));

llvm::cl::opt<std::string>
    OutputFilename("o", llvm::cl::desc("path for the output file"),
                   llvm::cl::init("a.out"), llvm::cl::cat(CLICat));

int main(int argc, char *argv[]) {
  llvm::cl::HideUnrelatedOptions(CLICat);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Chira CLI - MLIR-based Scheme compiler and runtime\n");

  mlir::MLIRContext context;

  auto input_res = chira::parser::Input::FromFile(InputFilename);
  if (!input_res) {
    llvm::errs() << "error: " << input_res.takeError() << "\n";
    return 1;
  }
  auto input = *input_res;

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

  chira::parser::Tokenizer tokenizer(context, input);

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

  mlir::PassManager pm(module->getContext());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  if (mlir::failed(pm.run(module))) {
    return 1;
  }

  module->print(os, flags);
  return 0;
}
