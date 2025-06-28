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

#ifndef CHIRA_RUNTIME_CHIRART
#define CHIRA_RUNTIME_CHIRART

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

inline namespace chirart {

template <typename... Args>
[[gnu::always_inline]] void assert(bool cond, const char *msg, Args &&...args) {
  if (!cond) [[unlikely]] {
    fprintf(stderr, "Assertion failed: ");
    fprintf(stderr, msg, args...);
    fprintf(stderr, "\n");
    std::abort();
  }
}

[[gnu::always_inline]] [[noreturn]] inline void unreachable(const char *msg) {
  fprintf(stderr, "Unreachable: %s\n", msg);
  std::abort();
}

struct Var;
struct ArgList;

using Args = ArgList *;
using Env = Var **;
using Lambda = void (*)(Var *, Args, Env);

inline struct Nil {
} nil;

struct Var {
private:
  enum Tag : size_t {
    UNSPEC = 0, // nil
    INT = 1,    // { i64 }
    FLOAT = 2,  // { f64 }
    BOOL = 3,   // { bool }
    STRING = 4, // { data ptr, size i64 }
    PAIR = 5,   // { left ptr, right ptr }
    NIL = 6,    // nil

    // param_size: 1bit flag (param_size >> 15) | 15bit size (param_size &
    // 0x7fff), if flag == 0, the provided arg size must equal to size if flag
    // == 1, the provided arg size must equal to or larger than size
    PRIM_OP_BEGIN =
        1 << 16, // PRIM_OP_BEGIN + param_size, { func ptr, null ptr }
    PRIM_OP_END = PRIM_OP_BEGIN + (1 << 16),
    CLOSURE_BEGIN =
        2 << 16, // CLOSURE_BEGIN + param_size, { func ptr, env ptr }
    CLOSURE_END = CLOSURE_BEGIN + (1 << 16),
  } tag;

  union {
    int64_t int_;
    double float_;
    bool bool_;
    struct {
      Lambda lambda;
      Env env;
    } closure;
    struct {
      char *data;
      size_t size;
    } string;
    struct {
      Var *left;
      Var *right;
    } pair;
    struct {
      uint64_t a;
      uint64_t b;
    } underlying;
  } data;

  [[gnu::always_inline]] void copyData(const Var &other) {
    data.underlying.a = other.data.underlying.a;
    data.underlying.b = other.data.underlying.b;
  }

public:
  [[gnu::always_inline]] Var() : tag(UNSPEC) {}
  [[gnu::always_inline]] Var(int64_t v) : tag(INT) { data.int_ = v; }
  [[gnu::always_inline]] Var(double v) : tag(FLOAT) { data.float_ = v; }
  [[gnu::always_inline]] Var(bool v) : tag(BOOL) { data.bool_ = v; }
  [[gnu::always_inline]] Var(Lambda lambda, Env env, size_t param_size)
      : tag(Tag(CLOSURE_BEGIN + param_size)) {
    assert(param_size < (1 << 16), "Too many closure captures");
    data.closure.lambda = lambda;
    data.closure.env = env;
  }
  [[gnu::always_inline]] Var(void *func_ptr, size_t param_size)
      : tag(Tag(PRIM_OP_BEGIN + param_size)) {
    assert(param_size < (1 << 16), "Too many closure captures");
    data.closure.lambda = (Lambda)func_ptr;
    data.closure.env = nullptr;
  }
  [[gnu::always_inline]] Var(char *data_ptr, size_t size) : tag(STRING) {
    data.string.data = data_ptr;
    data.string.size = size;
  }
  [[gnu::always_inline]] Var(Var *left, Var *right) : tag(PAIR) {
    data.pair.left = left;
    data.pair.right = right;
  }
  [[gnu::always_inline]] Var(Nil) : tag(NIL) {}

  [[gnu::always_inline]] Var(const Var &other) : tag(other.tag) {
    copyData(other);
  }

  [[gnu::always_inline]] Var &operator=(const Var &other) {
    tag = other.tag;
    copyData(other);
    return *this;
  }

  [[gnu::always_inline]] bool isUnspecified() const { return tag == UNSPEC; }
  [[gnu::always_inline]] bool isInt() const { return tag == INT; }
  [[gnu::always_inline]] bool isFloat() const { return tag == FLOAT; }
  [[gnu::always_inline]] bool isBool() const { return tag == BOOL; }
  [[gnu::always_inline]] bool isClosure() const {
    return tag >= CLOSURE_BEGIN && tag < CLOSURE_END;
  }
  [[gnu::always_inline]] bool isString() const { return tag == STRING; }
  [[gnu::always_inline]] bool isPrimOp() const {
    return tag >= PRIM_OP_BEGIN && tag < PRIM_OP_END;
  }
  [[gnu::always_inline]] bool isPair() const { return tag == PAIR; }
  [[gnu::always_inline]] bool isNil() const { return tag == NIL; }

  [[gnu::always_inline]] int64_t getInt() const {
    assert(isInt(), "Var is not an integer");
    return data.int_;
  }

  [[gnu::always_inline]] double getFloat() const {
    assert(isFloat() || isInt(),
           "Var is not a floating point number or integer");
    if (isFloat())
      return data.float_;
    else
      return data.int_;
  }

  [[gnu::always_inline]] bool getBool() const {
    assert(isBool(), "Var is not a boolean");
    return data.bool_;
  }

  [[gnu::always_inline]] Lambda getLambda() const {
    assert(isClosure() || isPrimOp(),
           "Var is not a closure or primary operation");
    return data.closure.lambda;
  }

  [[gnu::always_inline]] Env getEnv() const {
    assert(isClosure() || isPrimOp(),
           "Var is not a closure or primary operation");
    return data.closure.env;
  }

  [[gnu::always_inline]] size_t getParamSize() const {
    assert(isClosure() || isPrimOp(),
           "Var is not a closure or primary operation");

    return tag - (isClosure() ? CLOSURE_BEGIN : PRIM_OP_BEGIN);
  }

  [[gnu::always_inline]] char *getStringData() const {
    assert(isString(), "Var is not a string");
    return data.string.data;
  }

  [[gnu::always_inline]] size_t getStringSize() const {
    assert(isString(), "Var is not a string");
    return data.string.size;
  }

  [[gnu::always_inline]] Var *getLeft() const {
    assert(isPair(), "Var is not a pair");
    return data.pair.left;
  }

  [[gnu::always_inline]] Var *getRight() const {
    assert(isPair(), "Var is not a pair");
    return data.pair.right;
  }

  [[gnu::always_inline]] friend Var operator+(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() + r.getInt());
    } else if ((l.isFloat() || l.isInt()) && (r.isFloat() || r.isInt())) {
      return Var(l.getFloat() + r.getFloat());
    }

    unreachable("Invalid type to perform addition");
  }

  [[gnu::always_inline]] friend Var operator-(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() - r.getInt());
    } else if ((l.isFloat() || l.isInt()) && (r.isFloat() || r.isInt())) {
      return Var(l.getFloat() - r.getFloat());
    }

    unreachable("Invalid type to perform subtraction");
  }

  [[gnu::always_inline]] friend Var operator*(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() * r.getInt());
    } else if ((l.isFloat() || l.isInt()) && (r.isFloat() || r.isInt())) {
      return Var(l.getFloat() * r.getFloat());
    }

    unreachable("Invalid type to perform multiplication");
  }

  [[gnu::always_inline]] friend Var operator/(const Var &l, const Var &r) {
    if ((l.isFloat() || l.isInt()) && (r.isFloat() || r.isInt())) {
      assert(r.getFloat() != 0, "Division by zero");
      return Var(l.getFloat() / r.getFloat());
    }

    unreachable("Invalid type to perform division");
  }

  [[gnu::always_inline]] friend Var operator<(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() < r.getInt());
    } else if ((l.isFloat() || l.isInt()) && (r.isFloat() || r.isInt())) {
      return Var(l.getFloat() < r.getFloat());
    }

    unreachable("Invalid type to perform < comparison");
  }

  [[gnu::always_inline]] friend Var operator<=(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() <= r.getInt());
    } else if ((l.isFloat() || l.isInt()) && (r.isFloat() || r.isInt())) {
      return Var(l.getFloat() <= r.getFloat());
    }

    unreachable("Invalid type to perform <= comparison");
  }

  [[gnu::always_inline]] friend Var operator>(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() > r.getInt());
    } else if ((l.isFloat() || l.isInt()) && (r.isFloat() || r.isInt())) {
      return Var(l.getFloat() > r.getFloat());
    }

    unreachable("Invalid type to perform > comparison");
  }

  [[gnu::always_inline]] friend Var operator>=(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() >= r.getInt());
    } else if ((l.isFloat() || l.isInt()) && (r.isFloat() || r.isInt())) {
      return Var(l.getFloat() >= r.getFloat());
    }

    unreachable("Invalid type to perform >= comparison");
  }

  [[gnu::always_inline]] friend Var operator==(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() == r.getInt());
    } else if ((l.isFloat() || l.isInt()) && (r.isFloat() || r.isInt())) {
      return Var(l.getFloat() == r.getFloat());
    }

    unreachable("Invalid type to perform numeric equality check");
  }

  [[gnu::always_inline]] static Var Eq(const Var &l, const Var &r) {
    if ((l.isFloat() || l.isInt()) && (r.isFloat() || r.isInt())) {
      return l == r;
    } else if (l.isBool() && r.isBool()) {
      return Var(l.getBool() == r.getBool());
    } else if (l.isNil() && r.isNil()) {
      return Var(true);
    } else if (l.isString() && r.isString()) {
      return Var(l.getStringSize() == r.getStringSize() &&
                 l.getStringData() == r.getStringData());
    } else if (l.isPair() && r.isPair()) {
      return Var(l.getLeft() == r.getLeft() && l.getRight() == r.getRight());
    } else if (l.isUnspecified() && r.isUnspecified()) {
      return Var(false);
    }

    unreachable("Invalid type to perform shallow equality check");
  }

  [[gnu::always_inline]] static Var Equal(const Var &l, const Var &r) {
    if ((l.isFloat() || l.isInt()) && (r.isFloat() || r.isInt())) {
      return l == r;
    } else if (l.isBool() && r.isBool()) {
      return Var(l.getBool() == r.getBool());
    } else if (l.isNil() && r.isNil()) {
      return Var(true);
    } else if (l.isString() && r.isString()) {
      return Var(
          l.getStringSize() == r.getStringSize() &&
          memcmp(l.getStringData(), r.getStringData(), l.getStringSize()) == 0);
    } else if (l.isPair() && r.isPair()) {
      return Var(Equal(*l.getLeft(), *r.getLeft()) &&
                 Equal(*l.getRight(), *r.getRight()));
    } else if (l.isUnspecified() && r.isUnspecified()) {
      return Var(false);
    }

    unreachable("Invalid type to perform deep equality check");
  }

  [[gnu::always_inline]] Var operator!() {
    if (isBool()) {
      return Var(!getBool());
    }

    unreachable("Invalid type to perform logical negation");
  }

  [[gnu::always_inline]] friend Var operator&&(const Var &l, const Var &r) {
    if (l.isBool() && r.isBool()) {
      return Var(l.getBool() && r.getBool());
    }

    unreachable("Invalid type to perform logical AND");
  }

  [[gnu::always_inline]] friend Var operator||(const Var &l, const Var &r) {
    if (l.isBool() && r.isBool()) {
      return Var(l.getBool() || r.getBool());
    }

    unreachable("Invalid type to perform logical OR");
  }

  [[gnu::always_inline]] void Display() const {
    if (isInt()) {
      fprintf(stdout, "%ld", getInt());
      return;
    } else if (isFloat()) {
      fprintf(stdout, "%lf", getFloat());
      return;
    } else if (isBool()) {
      fprintf(stdout, getBool() ? "#t" : "#f");
      return;
    }

    unreachable("Not implemented yet");
  }

  [[gnu::always_inline]] inline Var operator()(Args args);
};

static_assert(sizeof(Var) == 3 * sizeof(uint64_t));

struct ArgList {
  size_t size;
  Var args[];

  Var *begin() { return args; }
  Var *end() { return args + size; }
};

inline constexpr const uint16_t PARAM_FLAG_BIT = 0x8000;
inline constexpr const uint16_t PARAM_VAL_MASK = 0x7fff;

[[gnu::always_inline]] inline uint16_t MakeParamSize(bool flag, size_t size) {
  assert(size < (1 << 15), "Parameter size too large");
  return (flag ? PARAM_FLAG_BIT : 0) | (size & PARAM_VAL_MASK);
}

[[gnu::always_inline]] inline Var Var::operator()(Args args) {
  auto param_size = getParamSize();

  auto expected_size = param_size & PARAM_VAL_MASK;
  if ((param_size & PARAM_FLAG_BIT) == 0) {
    assert(args->size == expected_size,
           "Argument size mismatch (expected %zu, got %zu)", expected_size,
           args->size);
  } else {
    assert(args->size >= expected_size,
           "Argument size mismatch (expected no less than %zu, got %zu)",
           expected_size, args->size);
  }

  Var res;
  getLambda()(&res, args, getEnv());
  return res;
}

[[gnu::always_inline]] inline void Newline() { fprintf(stdout, "\n"); }

} // namespace chirart

#endif // CHIRA_RUNTIME_CHIRART
