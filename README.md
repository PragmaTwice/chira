# Chira

**Chira** (pronounced /ˈkaɪ.rə/) is an MLIR-based optimizing JIT compiler and runtime for the Scheme programming language.

## Status

Work in progress to be R7RS-small compatible.

## Compilation pipeline

```
Source Code
  ↓
SExpr (S-expression syntax tree, for macro expansion)
  ↓
SIR (Scheme IR, in a high-level ANF form)
  ↓
CHIR + Upstream Dialects (SSA-based middle-level IR)
  ↓
LLVM IR (target-independent low-level IR)
  ↓
Native Code
```
