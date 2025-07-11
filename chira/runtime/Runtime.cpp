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

#include "chira/runtime/Runtime.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>

namespace chira::rt {

static constexpr const char ChirartCode[] = {
#include "chira/runtime/chirart.ll.inc"
    , 0};

static constexpr const char ChirartMainCode[] = {
#include "chira/runtime/chirart_main.ll.inc"
    , 0};

static std::unique_ptr<llvm::Module> createModule(llvm::LLVMContext &ctx,
                                                  llvm::StringRef code,
                                                  llvm::StringRef name) {
  auto buffer = llvm::MemoryBuffer::getMemBuffer(code, name);

  llvm::SMDiagnostic err;
  auto module = llvm::parseIR(buffer->getMemBufferRef(), err, ctx);
  if (!module) {
    err.print("", llvm::errs());
    std::exit(1);
  }

  return module;
}

std::unique_ptr<llvm::Module> createRuntimeModule(llvm::LLVMContext &ctx) {
  return createModule(ctx, llvm::StringRef(ChirartCode, sizeof ChirartCode - 1),
                      "chirart.ll");
}

std::unique_ptr<llvm::Module>
createRuntimeLibcMainModule(llvm::LLVMContext &ctx) {
  return createModule(
      ctx, llvm::StringRef(ChirartMainCode, sizeof ChirartMainCode - 1),
      "chirart_main.ll");
}

} // namespace chira::rt
