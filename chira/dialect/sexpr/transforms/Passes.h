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

#ifndef CHIRA_DIALECT_SEXPR_TRANSFORMS_PASSES
#define CHIRA_DIALECT_SEXPR_TRANSFORMS_PASSES

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace chira::sexpr {

struct ConversionContext {
  bool has_error = false;

  void SetError() { has_error = true; }
  bool Good() { return !has_error; }
};

template <typename Op> struct RewritePattern : mlir::OpRewritePattern<Op> {
  RewritePattern(mlir::MLIRContext *ctx, ConversionContext &cvt_ctx)
      : mlir::OpRewritePattern<Op>(ctx), cvt_ctx(cvt_ctx) {}

  auto emitError(mlir::Location loc) const {
    cvt_ctx.SetError();
    return mlir::emitError(loc);
  }

  ConversionContext &cvt_ctx;
};

std::unique_ptr<mlir::Pass> createDefineSyntaxConstructorPass();

std::unique_ptr<mlir::Pass> createMacroExpanderPass();

} // namespace chira::sexpr

#endif // CHIRA_DIALECT_SEXPR_TRANSFORMS_PASSESF
