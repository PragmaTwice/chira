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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

inline namespace chirart {

using Lambda = void *;

struct Var;
using Env = Var **;

inline void *GCMalloc(size_t n) { return malloc(n); }

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

  void copyData(const Var &other) {
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
      assert(false && "Invalid Var tag");
    }
  }

public:
  Var() : tag(UNSPEC) {}
  Var(int64_t v) : tag(INT) { data.int_ = v; }
  Var(double v) : tag(FLOAT) { data.float_ = v; }
  Var(bool v) : tag(BOOL) { data.bool_ = v; }
  Var(Lambda func_ptr, Env caps, size_t cap_size)
      : tag(Tag(CLOSURE_BEGIN + cap_size)) {
    assert(cap_size < (1 << 16) && "Too many closure captures");
    data.closure.func_ptr = func_ptr;
    data.closure.caps = caps;
  }
  Var(Lambda func_ptr) : Var(func_ptr, nullptr, 0) {}

  Var(const Var &other) : tag(other.tag) { copyData(other); }

  Var &operator=(const Var &other) {
    tag = other.tag;
    copyData(other);
    return *this;
  }

  bool isUnspecified() const { return tag == UNSPEC; }
  bool isInt() const { return tag == INT; }
  bool isFloat() const { return tag == FLOAT; }
  bool isBool() const { return tag == BOOL; }
  bool isClosure() const { return tag >= CLOSURE_BEGIN && tag < CLOSURE_END; }

  int64_t getInt() const {
    assert(isInt() && "Var is not an integer");
    return data.int_;
  }

  double getFloat() const {
    assert(isFloat() && "Var is not a float");
    return data.float_;
  }

  bool getBool() const {
    assert(isBool() && "Var is not a boolean");
    return data.bool_;
  }

  Lambda getFuncPtr() const {
    assert(isClosure() && "Var is not a closure");
    return data.closure.func_ptr;
  }

  Env getCaps() const {
    assert(isClosure() && "Var is not a closure");
    return data.closure.caps;
  }

  size_t getCapSize() const {
    assert(isClosure() && "Var is not a closure");
    return tag - CLOSURE_BEGIN;
  }

  template <typename... Args> static Var *Create(Args... args) {
    auto var = static_cast<Var *>(GCMalloc(sizeof(Var)));
    new (var) Var(args...);
    return var;
  }

  friend Var operator+(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() + r.getInt());
    }

    assert(false && "Not implemented yet");
  }

  friend Var operator-(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() - r.getInt());
    }

    assert(false && "Not implemented yet");
  }

  friend Var operator<(const Var &l, const Var &r) {
    if (l.isInt() && r.isInt()) {
      return Var(l.getInt() < r.getInt());
    }

    assert(false && "Not implemented yet");
  }

  void Display() const {
    if (isInt()) {
      printf("%ld", getInt());
      return;
    }

    assert(false && "Not implemented yet");
  }
};

static_assert(sizeof(Var) == 24);

} // namespace chirart

extern "C" {

Var *chirart_unspec() { return Var::Create(); }
Var *chirart_int(int64_t num) { return Var::Create(num); }
Var *chirart_closure(Lambda func_ptr, Env caps, size_t cap_size) {
  return Var::Create(func_ptr, caps, cap_size);
}
Var *chirart_closure_nocap(Lambda func_ptr) { return Var::Create(func_ptr); }
void chirart_set(Var *l, Var *r) { *l = *r; }
Env chirart_env(size_t size) {
  return static_cast<Env>(GCMalloc(size * sizeof(Var *)));
}
Lambda chirart_get_func_ptr(Var *v) { return v->getFuncPtr(); }
Env chirart_get_caps(Var *v) { return v->getCaps(); }
Var *chirart_env_load(Env env, size_t idx) { return env[idx]; }
void chirart_env_store(Env env, size_t idx, Var *v) { env[idx] = v; }
Var *chirart_copy(Var *v) { return Var::Create(*v); }
bool chirart_get_bool(Var *v) { return v->getBool(); }
Var *chirart_add(Var *l, Var *r) { return Var::Create(*l + *r); }
Var *chirart_subtract(Var *l, Var *r) { return Var::Create(*l - *r); }
Var *chirart_lt(Var *l, Var *r) { return Var::Create(*l < *r); }
Var *chirart_display(Var *l) {
  l->Display();
  return chirart_unspec();
}

Var *chiracg_main(Env);
}

int main() {
  chiracg_main(nullptr);
}
