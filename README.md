# Chira

**Chira** (pronounced /ˈkaɪ.rə/) is an MLIR-based high-performance compiler and runtime for the Scheme programming language.

## Status

Work in progress to be R7RS-small compatible.

## Design

planned IR levels:
- sexpr: S-expression syntax tree IR (macro expansion)
- sir: Scheme high-level ANF-based IR
- chir + MLIR dialects: middle-level SSA-based IR
- LLVM IR: target IR
