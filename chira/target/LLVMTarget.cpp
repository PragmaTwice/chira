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

#include "chira/target/LLVMTarget.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

namespace chira::target {

std::unique_ptr<llvm::Module> translateToLLVM(mlir::ModuleOp module,
                                              llvm::LLVMContext &ctx,
                                              llvm::StringRef name) {
  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerBuiltinDialectTranslation(*module.getContext());

  return mlir::translateModuleToLLVMIR(module.getOperation(), ctx, name);
}

void optimizeLLVMModule(llvm::Module &module) {
  llvm::PassBuilder builder;

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  builder.registerModuleAnalyses(mam);
  builder.registerCGSCCAnalyses(cgam);
  builder.registerFunctionAnalyses(fam);
  builder.registerLoopAnalyses(lam);
  builder.crossRegisterProxies(lam, fam, cgam, mam);

  auto llvm_manager =
      builder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
  llvm_manager.run(module, mam);
}

llvm::Error emitObjectFile(llvm::Module &module, llvm::raw_pwrite_stream &os) {
  using namespace llvm;

  std::string target_triple = sys::getDefaultTargetTriple();
  module.setTargetTriple(target_triple);

  std::string error;
  const Target *target = TargetRegistry::lookupTarget(target_triple, error);
  if (!target) {
    return make_error<StringError>("failed to lookup target: " + error,
                                   inconvertibleErrorCode());
  }

  TargetOptions opt;
  auto rm = std::optional<Reloc::Model>();
  std::unique_ptr<TargetMachine> target_machine(
      target->createTargetMachine(target_triple, "generic", "", opt, rm));
  if (!target_machine) {
    return make_error<StringError>("failed to create target machine",
                                   inconvertibleErrorCode());
  }

  module.setDataLayout(target_machine->createDataLayout());

  legacy::PassManager pass;
  if (target_machine->addPassesToEmitFile(pass, os, nullptr,
                                          CodeGenFileType::ObjectFile)) {
    return make_error<StringError>("failed to emit object file",
                                   inconvertibleErrorCode());
  }

  pass.run(module);
  os.flush();

  return Error::success();
}

} // namespace hyperbrain::target
