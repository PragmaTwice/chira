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

#include "chirart.h"

extern "C" {
void chirart_unspec(Var *r) { *r = Var(); }

void chirart_int(Var *r, int64_t num) { *r = Var(num); }
void chirart_float(Var *r, double num) { *r = Var(num); }
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
void chirart_sub(Var *v, const Var *l, const Var *r) { *v = *l - *r; }
void chirart_mul(Var *v, const Var *l, const Var *r) { *v = *l * *r; }
void chirart_div(Var *v, const Var *l, const Var *r) { *v = *l / *r; }
void chirart_lt(Var *v, const Var *l, const Var *r) { *v = *l < *r; }
void chirart_le(Var *v, const Var *l, const Var *r) { *v = *l <= *r; }
void chirart_gt(Var *v, const Var *l, const Var *r) { *v = *l > *r; }
void chirart_ge(Var *v, const Var *l, const Var *r) { *v = *l >= *r; }
void chirart_eq(Var *v, const Var *l, const Var *r) { *v = *l == *r; }

void chirart_display(Var *v, const Var *l) {
  l->Display();
  chirart_unspec(v);
}
void chirart_newline(Var *v) {
  Newline();
  chirart_unspec(v);
}

void chiracg_main(Var *, Env);
}

int main() {
  Var r;
  chiracg_main(&r, nullptr);
}
