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

std::unique_ptr<llvm::Module> createRuntimeModule(llvm::LLVMContext &ctx) {
  auto chirart_buffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(ChirartCode, sizeof ChirartCode - 1), "chirart.ll");

  llvm::SMDiagnostic err;
  auto chirart = llvm::parseIR(chirart_buffer->getMemBufferRef(), err, ctx);
  if (!chirart) {
    err.print("", llvm::errs());
    std::exit(1);
  }

  return chirart;
}

} // namespace chira::rt
