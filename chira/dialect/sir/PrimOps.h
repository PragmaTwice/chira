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

#ifndef CHIRA_DIALECT_SIR_PRIMOPS
#define CHIRA_DIALECT_SIR_PRIMOPS

#include "llvm/ADT/StringRef.h"
#include <map>
#include <vector>

namespace chira::sir {

inline struct NoLessThan {
} no_less_than;

struct Arity {
  bool flag;
  size_t size;

  Arity(size_t size) : flag(false), size(size) {}
  Arity(NoLessThan, size_t size) : flag(true), size(size) {}

  inline static constexpr const uint16_t PARAM_FLAG_BIT = 0x8000;
  inline static constexpr const uint16_t PARAM_VAL_MASK = 0x7fff;

  size_t Encoded() const {
    assert(size < (1 << 15) && "Parameter size too large");
    return (flag ? PARAM_FLAG_BIT : 0) | (size & PARAM_VAL_MASK);
  }
};

struct PrimOp {
  enum Kind { Arith, IO };

  llvm::StringRef symbol;
  llvm::StringRef name;
  Kind kind;
  Arity num_args;

  PrimOp(llvm::StringRef symbol, llvm::StringRef name, Kind kind,
         Arity num_args)
      : symbol(symbol), name(name), kind(kind), num_args(num_args) {}
};

struct PrimOps {
  static inline std::vector<PrimOp> list = {
      {"+", "add", PrimOp::Arith, {no_less_than, 1}},
      {"-", "sub", PrimOp::Arith, 2},
      {"*", "mul", PrimOp::Arith, 2},
      {"/", "div", PrimOp::Arith, 2},
      {"<", "lt", PrimOp::Arith, 2},
      {">", "gt", PrimOp::Arith, 2},
      {"<=", "le", PrimOp::Arith, 2},
      {">=", "ge", PrimOp::Arith, 2},
      {"=", "eq", PrimOp::Arith, 2},
      {"not", "not", PrimOp::Arith, 1},
      {"and", "and", PrimOp::Arith, 2},
      {"or", "or", PrimOp::Arith, 2},
      {"display", "display", PrimOp::IO, 1},
      {"newline", "newline", PrimOp::IO, 0}};

  static inline std::map<llvm::StringRef, PrimOp> map_by_symbol = [] {
    std::map<llvm::StringRef, PrimOp> map;
    for (const auto &op : list) {
      map.emplace(op.symbol, op);
    }

    return map;
  }();

  static inline std::map<llvm::StringRef, PrimOp> map_by_name = [] {
    std::map<llvm::StringRef, PrimOp> map;
    for (const auto &op : list) {
      map.emplace(op.name, op);
    }

    return map;
  }();

  static std::optional<PrimOp> getBySymbol(llvm::StringRef symbol) {
    if (auto it = map_by_symbol.find(symbol); it != map_by_symbol.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  static std::optional<PrimOp> getByName(llvm::StringRef name) {
    if (auto it = map_by_name.find(name); it != map_by_name.end()) {
      return it->second;
    }
    return std::nullopt;
  }
};

} // namespace chira::sir

#endif // CHIRA_DIALECT_SIR_PRIMOPS
