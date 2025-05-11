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

#ifndef CHIRA_PARSER_INPUT
#define CHIRA_PARSER_INPUT

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace chira::parser {

struct Input {
  llvm::StringRef Filename() const { return filename; }
  llvm::StringRef Content() const { return content; }

  static Input FromMemory(llvm::StringRef content) {
    return Input("<memory>", content);
  }

  static llvm::Expected<Input> FromFile(llvm::StringRef filename) {
    auto result = llvm::MemoryBuffer::getFile(filename, true);
    if (!result)
      return llvm::createStringError(result.getError(), "failed to read file");

    return Input(filename, result->get()->getBuffer());
  }

private:
  std::string filename;
  std::string content;

  Input(llvm::StringRef filename, llvm::StringRef content)
      : filename(filename), content(content){};
};

} // namespace chira::parser

#endif // CHIRA_PARSER_INPUT
