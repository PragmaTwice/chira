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
#include "chira/dialect/sir/transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"

namespace chira::sir {

namespace {

struct RecursiveResolverPass
    : public mlir::PassWrapper<RecursiveResolverPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<mlir::Pass> createRecursiveResolverPass() {
  return std::make_unique<RecursiveResolverPass>();
}

} // namespace chira::sir
