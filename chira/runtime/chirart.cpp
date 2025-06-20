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

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

inline namespace chirart {

[[gnu::always_inline]] inline void assert(bool cond, const char *msg) {
  if (!cond) [[unlikely]] {
    fprintf(stderr, "Assertion failed: %s\n", msg);
    std::abort();
  }
}

[[gnu::always_inline]] [[noreturn]] inline void unreachable(const char *msg) {
  fprintf(stderr, "Unreachable: %s\n", msg);
  std::abort();
}

using Lambda = void *;

struct Var;
using Env = Var **;

struct Var {
private:
  enum Tag : size_t {
    UNSPEC = 0,
    INT = 1,
    FLOAT = 2,
    BOOL = 3,

    CLOSURE_BEGIN = 1 << 16,
    CLOSURE_END = CLOSURE_BEGIN + (1 << 16),
  } tag;

  union {
    int64_t int_;
    double float_;
    bool bool_;
    struct {
      Lambda func_ptr;
      Env caps;
    } closure;
  } data;

  [[gnu::always_inline]] void copyData(const Var &other) {
    if (tag == INT) {
      data.int_ = other.data.int_;
    } else if (tag == FLOAT) {
      data.float_ = other.data.float_;
    } else if (tag == BOOL) {
      data.bool_ = other.data.bool_;
    } else if (tag >= CLOSURE_BEGIN && tag < CLOSURE_END) {
      data.closure.func_ptr = other.data.closure.func_ptr;
      data.closure.caps = other.data.closure.caps;
    } else if (tag == UNSPEC) {
      // nothing to copy for UNSPEC
    } else {
      assert(false, "Invalid tag in Var");
    }
  }

public:
  [[gnu::always_inline]] Var() : tag(UNSPEC) {}
  [[gnu::always_inline]] Var(int64_t v) : tag(INT) { data.int_ = v; }
  [[gnu::always_inline]] Var(double v) : tag(FLOAT) { data.float_ = v; }
  [[gnu::always_inline]] Var(bool v) : tag(BOOL) { data.bool_ = v; }
  [[gnu::always_inline]] Var(Lambda func_ptr, Env caps, size_t cap_size)
      : tag(Tag(CLOSURE_BEGIN + cap_size)) {
    assert(cap_size < (1 << 16), "Too many closure captures");
    data.closure.func_ptr = func_ptr;
    data.closure.caps = caps;
  }
  [[gnu::always_inline]] Var(Lambda func_ptr) : Var(func_ptr, nullptr, 0) {}

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

  [[gnu::always_inline]] int64_t getInt() const {
    assert(isInt(), "Var is not an integer");
    return data.int_;
  }

  [[gnu::always_inline]] double getFloat() const {
    assert(isFloat(), "Var is not a float");
    return data.float_;
  }

  [[gnu::always_inline]] bool getBool() const {
    assert(isBool(), "Var is not a boolean");
    return data.bool_;
  }

  [[gnu::always_inline]] Lambda getFuncPtr() const {
    assert(isClosure(), "Var is not a closure");
    return data.closure.func_ptr;
  }

  [[gnu::always_inline]] Env getCaps() const {
    assert(isClosure(), "Var is not a closure");
    return data.closure.caps;
  }

  [[gnu::always_inline]] size_t getCapSize() const {
    assert(isClosure(), "Var is not a closure");
    return tag - CLOSURE_BEGIN;
  }

  [[gnu::always_inline]] friend Var operator+(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() + r.getInt());
    }

    unreachable("Not implemented yet");
  }

  [[gnu::always_inline]] friend Var operator-(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() - r.getInt());
    }

    unreachable("Not implemented yet");
  }

  [[gnu::always_inline]] friend Var operator<(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() < r.getInt());
    }

    unreachable("Not implemented yet");
  }

  [[gnu::always_inline]] void Display() const {
    if (isInt()) {
      fprintf(stdout, "%ld", getInt());
      return;
    }

    unreachable("Not implemented yet");
  }
};

static_assert(sizeof(Var) == 24);

} // namespace chirart

extern "C" {

void chirart_unspec(Var *r) { *r = Var(); }
void chirart_int(Var *r, int64_t num) { *r = Var(num); }
void chirart_closure(Var *r, Lambda func_ptr, Env caps, size_t cap_size) {
  *r = Var(func_ptr, caps, cap_size);
}
void chirart_set(Var *l, const Var *r) { *l = *r; }
Lambda chirart_get_func_ptr(const Var *v) { return v->getFuncPtr(); }
Env chirart_get_caps(const Var *v) { return v->getCaps(); }
Var *chirart_env_load(Env env, size_t idx) { return env[idx]; }
void chirart_env_store(Env env, size_t idx, Var *v) { env[idx] = v; }
bool chirart_get_bool(const Var *v) { return v->getBool(); }
void chirart_add(Var *v, const Var *l, const Var *r) { *v = *l + *r; }
void chirart_subtract(Var *v, const Var *l, const Var *r) { *v = *l - *r; }
void chirart_lt(Var *v, const Var *l, const Var *r) { *v = *l < *r; }
void chirart_display(Var *v, const Var *l) {
  l->Display();
  chirart_unspec(v);
}

void chiracg_main(Var *, Env);
}

int main() {
  Var r;
  chiracg_main(&r, nullptr);
}
