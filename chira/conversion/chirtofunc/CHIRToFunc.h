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

#ifndef CHIRA_CONVERSION_CHIRTOFUNC_CHIRTOFUNC
#define CHIRA_CONVERSION_CHIRTOFUNC_CHIRTOFUNC

#include "mlir/Pass/Pass.h"

namespace chira {

std::unique_ptr<mlir::Pass> createCHIRToFuncConversionPass();

}

#endif // CHIRA_CONVERSION_CHIRTOFUNC_CHIRTOFUNC
