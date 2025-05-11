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

#ifndef CHIRA_TARGET_LLVMTARGET
#define CHIRA_TARGET_LLVMTARGET

#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace chira::target {

std::unique_ptr<llvm::Module> translateToLLVM(mlir::ModuleOp module,
                                              llvm::LLVMContext &ctx,
                                              llvm::StringRef name);

void optimizeLLVMModule(llvm::Module &module);

llvm::Error emitObjectFile(llvm::Module &module, llvm::raw_pwrite_stream &os);

} // namespace chira::target

#endif // CHIRA_TARGET_LLVMTARGET
