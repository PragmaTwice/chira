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

#ifndef CHIRA_RUNTIME_RUNTIME
#define CHIRA_RUNTIME_RUNTIME

#include "llvm/IR/Module.h"

namespace chira::rt {

std::unique_ptr<llvm::Module> createRuntimeModule(llvm::LLVMContext &ctx);
std::unique_ptr<llvm::Module> createRuntimeLibcMainModule(llvm::LLVMContext &ctx);

}

#endif // CHIRA_RUNTIME_RUNTIME
