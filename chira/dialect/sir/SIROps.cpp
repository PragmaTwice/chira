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

#include "chira/dialect/sir/SIROps.h"

#include "chira/dialect/sir/SIROpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "chira/dialect/sir/SIROpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "chira/dialect/sir/SIROps.cpp.inc"

namespace chira::sir {

void SIRDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "chira/dialect/sir/SIROpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "chira/dialect/sir/SIROps.cpp.inc"
      >();
}

} // namespace chira::sexpr
