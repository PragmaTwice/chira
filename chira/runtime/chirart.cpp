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
void chirart_closure(Var *r, Lambda lambda, Env env, size_t param_size) {
  *r = Var(lambda, env, param_size);
}
void chirart_prim_op(Var *r, void *func_ptr, size_t param_size) {
  *r = Var(func_ptr, param_size);
}

void chirart_set(Var *l, const Var *r) { *l = *r; }

Lambda chirart_get_lambda(const Var *v) { return v->getLambda(); }
Env chirart_get_env(const Var *v) { return v->getEnv(); }

Var *chirart_env_load(Env env, size_t idx) { return env[idx]; }
void chirart_env_store(Env env, size_t idx, Var *v) { env[idx] = v; }

void chirart_args_set_size(Args args, size_t size) { args->size = size; }
Var *chirart_args_load(Args args, size_t idx) { return &args->args[idx]; }
void chirart_args_store(Args args, size_t idx, Var *v) { args->args[idx] = *v; }

void chirart_call(Var *res, Var *callee, Args args) {
  assert(args->size == callee->getParamSize(), "Argument size mismatch");
  callee->getLambda()(res, args, callee->getEnv());
}

bool chirart_get_bool(const Var *v) { return v->getBool(); }

void chirart_add(Var *v, Args args, Env) {
  auto &l = args->args[0], &r = args->args[1];
  *v = l + r;
}
void chirart_sub(Var *v, Args args, Env) {
  auto &l = args->args[0], &r = args->args[1];
  *v = l - r;
}
void chirart_mul(Var *v, Args args, Env) {
  auto &l = args->args[0], &r = args->args[1];
  *v = l * r;
}
void chirart_div(Var *v, Args args, Env) {
  auto &l = args->args[0], &r = args->args[1];
  *v = l / r;
}
void chirart_lt(Var *v, Args args, Env) {
  auto &l = args->args[0], &r = args->args[1];
  *v = l < r;
}
void chirart_le(Var *v, Args args, Env) {
  auto &l = args->args[0], &r = args->args[1];
  *v = l <= r;
}
void chirart_gt(Var *v, Args args, Env) {
  auto &l = args->args[0], &r = args->args[1];
  *v = l > r;
}
void chirart_ge(Var *v, Args args, Env) {
  auto &l = args->args[0], &r = args->args[1];
  *v = l >= r;
}
void chirart_eq(Var *v, Args args, Env) {
  auto &l = args->args[0], &r = args->args[1];
  *v = l == r;
}

void chirart_display(Var *v, Args args, Env) {
  auto &l = args->args[0];
  l.Display();
  chirart_unspec(v);
}
void chirart_newline(Var *v, Args, Env) {
  Newline();
  chirart_unspec(v);
}

void chiracg_main(Var *, Args, Env);
}

int main() {
  Var r;
  chiracg_main(&r, nullptr, nullptr);
}
