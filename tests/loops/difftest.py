#!/usr/bin/env python3
"""Differential test: original C (gcc, x86) vs decompiled C (gcc, x86).

    python3 tests/loops/difftest.py [loops_O2.o ...]

For every object file the decompiled functions are renamed to <name>_dc,
compiled together with loops.c and a generated harness, and called on random
inputs; return values and written buffers must match.  Functions whose output
still contains __e2k_* / __asm__ are reported as "low-level" (not tested).
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from e2kdc.decompiler import Decompiler   # noqa: E402

FUNCS = ["for_up", "for_down", "for_step", "for_const", "for_ptr", "while_ptr", "while_cond",
         "do_while", "nested2", "for_break", "for_continue", "while_call", "inf_loop", "fill",
         "for_unsigned", "copy_arr"]

HARNESS = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef long (*F7)(long, long, long, long, long, long, long);
int ext(int x) { return x * 3 + 1; }
void take(int *p) { *p = *p * 5 + 2; }
%(protos)s
static unsigned long rs = 12345;
static long rnd(long lo, long hi) { rs = rs * 6364136223846793005UL + 1442695040888963407UL;
    return lo + (long)((rs >> 33) %% (unsigned long)(hi - lo + 1)); }
static int fails = 0, runs = 0;
#define CHK(name, a, b) do { runs++; if ((a) != (b)) { if (fails++ < 20) \
    printf("MISMATCH %%s: orig=%%ld dc=%%ld (case %%d)\n", name, (long)(a), (long)(b), it); } } while (0)
#define DC(f) ((F7)(void *)f##_dc)
/* float parameters travel in SSE registers on x86: call through the source prototype */
#define SAXPY_DC ((void (*)(int, float, const float *, float *))(void *)saxpy_dc)
int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  for (int it = 0; it < 400; it++) {
    int A[80], B[80]; long L[80]; char S[80], D1[80], D2[80];
    for (int i = 0; i < 80; i++) { A[i] = (int)rnd(-50, 50); B[i] = A[i]; L[i] = rnd(-1000, 1000); }
    int n = (int)rnd(-3, 60), m = (int)rnd(-2, 9), k = A[rnd(0, 70)];
    int slen = (int)rnd(0, 70);
    for (int i = 0; i < 80; i++) S[i] = i < slen ? (char)rnd(1, 120) : 0;
%(calls)s
  }
  printf("runs=%%d fails=%%d\n", runs, fails);
  return fails != 0;
}
'''

# name -> (C call of the original, C call of the decompiled version, comparison kind)
CASES = {
    "for_up":      "CHK(\"for_up\", for_up(A, n), (int)DC(for_up)((long)A, n, 0, 0, 0, 0, 0));",
    "for_down":    "CHK(\"for_down\", for_down(A, n), (int)DC(for_down)((long)A, n, 0, 0, 0, 0, 0));",
    "for_step":    "CHK(\"for_step\", for_step(A, n), (int)DC(for_step)((long)A, n, 0, 0, 0, 0, 0));",
    "for_const":   "CHK(\"for_const\", for_const(A), (int)DC(for_const)((long)A, 0, 0, 0, 0, 0, 0));",
    "for_ptr":     "{ int e = (int)rnd(0, 70); CHK(\"for_ptr\", for_ptr(L, L + e), DC(for_ptr)((long)L, (long)(L + e), 0, 0, 0, 0, 0)); }",
    "while_ptr":   "CHK(\"while_ptr\", while_ptr(S), (int)DC(while_ptr)((long)S, 0, 0, 0, 0, 0, 0));",
    "while_cond":  "{ unsigned x = (unsigned)rnd(0, 3000); CHK(\"while_cond\", while_cond(x), (int)DC(while_cond)((long)x, 0, 0, 0, 0, 0, 0)); }",
    "do_while":    "CHK(\"do_while\", do_while(n), (int)DC(do_while)(n, 0, 0, 0, 0, 0, 0));",
    "nested2":     "CHK(\"nested2\", nested2(n % 20, m), (int)DC(nested2)(n % 20, m, 0, 0, 0, 0, 0));",
    "for_break":   "CHK(\"for_break\", for_break(A, n, k), (int)DC(for_break)((long)A, n, k, 0, 0, 0, 0));",
    "for_continue": "CHK(\"for_continue\", for_continue(A, n), (int)DC(for_continue)((long)A, n, 0, 0, 0, 0, 0));",
    "while_call":  "CHK(\"while_call\", while_call(n), (int)DC(while_call)(n, 0, 0, 0, 0, 0, 0));",
    "inf_loop":    "{ int P[80]; for (int i = 0; i < 80; i++) P[i] = i < 60 ? (int)rnd(1, 3) : 0; P[0] = (int)rnd(0, 2);"
                   " CHK(\"inf_loop\", inf_loop(P), (int)DC(inf_loop)((long)P, 0, 0, 0, 0, 0, 0)); }",
    "fill":        "{ memset(D1, 7, 80); memset(D2, 7, 80); char c = (char)rnd(-100, 100); fill(D1, c, n);"
                   " DC(fill)((long)D2, c, n, 0, 0, 0, 0); CHK(\"fill\", memcmp(D1, D2, 80), 0); }",
    "for_unsigned": "{ unsigned u = (unsigned)rnd(0, 60); CHK(\"for_unsigned\", for_unsigned(u), (unsigned)DC(for_unsigned)(u, 0, 0, 0, 0, 0, 0)); }",
    "copy_arr":    "{ int X[80], Y[80]; for (int i = 0; i < 80; i++) X[i] = Y[i] = (int)rnd(-9, 9);"
                   " copy_arr(X, A, n); DC(copy_arr)((long)Y, (long)B, n, 0, 0, 0, 0);"
                   " CHK(\"copy_arr\", memcmp(X, Y, sizeof X), 0);"
                   # overlapping: dst = src +/- d exercises the MLOCK / check-load path
                   " int d = (int)rnd(1, 3); for (int i = 0; i < 80; i++) X[i] = Y[i] = (int)rnd(-9, 9);"
                   " int m2 = n < 70 ? n : 70; copy_arr(X + d, X, m2); DC(copy_arr)((long)(Y + d), (long)Y, m2, 0, 0, 0, 0);"
                   " CHK(\"copy_arr_overlap\", memcmp(X, Y, sizeof X), 0);"
                   " for (int i = 0; i < 80; i++) X[i] = Y[i] = (int)rnd(-9, 9);"
                   " copy_arr(X, X + d, m2); DC(copy_arr)((long)Y, (long)(Y + d), m2, 0, 0, 0, 0);"
                   " CHK(\"copy_arr_overlap_back\", memcmp(X, Y, sizeof X), 0); }",
}


ARCH_FUNCS = ["canteen", "canteen3", "spec_load", "wide", "saxpy", "dot", "fib", "clamp_abs", "find", "fsel"]
ARCH_CASES = {
    "canteen":   "CHK(\"canteen\", canteen(n & 1, k, m), (int)DC(canteen)(n & 1, k, m, 0, 0, 0, 0));",
    "canteen3":  "CHK(\"canteen3\", canteen3(n - 20, m - 3, k, A[3], A[4]), (int)DC(canteen3)(n - 20, m - 3, k, A[3], A[4], 0, 0));",
    "spec_load": "{ int *p = (n & 1) ? A + 5 : 0; CHK(\"spec_load\", spec_load(p, k), (int)DC(spec_load)((long)p, k, 0, 0, 0, 0, 0)); }",
    "wide":      "CHK(\"wide\", wide(L[0], L[1], L[2], L[3], L[4], L[5]), DC(wide)(L[0], L[1], L[2], L[3], L[4], L[5], 0));",
    "saxpy":     "{ float x[80], y1[80], y2[80]; for (int i = 0; i < 80; i++) { x[i] = (float)A[i] / 4; y1[i] = y2[i] = (float)B[i] / 8; }"
                 " float a = (float)k / 3; int ab; memcpy(&ab, &a, 4); saxpy(n, a, x, y1);"
                 " SAXPY_DC(n, a, x, y2); CHK(\"saxpy\", memcmp(y1, y2, sizeof y1), 0);"
                 " int d = (int)rnd(-3, 3), m2 = n < 70 ? n : 70; float z1[90], z2[90];"
                 " for (int i = 0; i < 90; i++) z1[i] = z2[i] = (float)A[i % 80] / 5;"
                 " saxpy(m2, a, z1 + 4 + d, z1 + 4); SAXPY_DC(m2, a, z2 + 4 + d, z2 + 4);"
                 " CHK(\"saxpy_overlap\", memcmp(z1, z2, sizeof z1), 0); }",
    "dot":       "CHK(\"dot\", dot(A, B, n), DC(dot)((long)A, (long)B, n, 0, 0, 0, 0));",
    "fib":       "CHK(\"fib\", fib(n % 22), (int)DC(fib)(n % 22, 0, 0, 0, 0, 0, 0));",
    "clamp_abs": "CHK(\"clamp_abs\", clamp_abs(k, m, n), (int)DC(clamp_abs)(k, m, n, 0, 0, 0, 0));",
    "find":      "CHK(\"find\", find(A, n, k), (int)DC(find)((long)A, n, k, 0, 0, 0, 0));",
    "fsel":      "{ double x = A[1] / 3.0, y = A[2] / 7.0; double r1 = fsel(x, y), r2 = ((double (*)(double, double))(void *)fsel_dc)(x, y);"
                 " CHK(\"fsel\", memcmp(&r1, &r2, 8), 0); }",
}
T1_FUNCS = ["add", "mul3", "max", "absdiff", "sum", "glob_inc", "call2", "sp", "fact", "sw", "fma3",
            "copy", "strlen2", "bits"]
T1_CASES = {
    "add":      "CHK(\"add\", add(k, n), (int)DC(add)(k, n, 0, 0, 0, 0, 0));",
    "mul3":     "CHK(\"mul3\", mul3(L[0], L[1], L[2]), DC(mul3)(L[0], L[1], L[2], 0, 0, 0, 0));",
    "max":      "CHK(\"max\", max(k, n), (int)DC(max)(k, n, 0, 0, 0, 0, 0));",
    "absdiff":  "CHK(\"absdiff\", absdiff(k, n), (int)DC(absdiff)(k, n, 0, 0, 0, 0, 0));",
    "sum":      "CHK(\"sum\", sum(A, n), (int)DC(sum)((long)A, n, 0, 0, 0, 0, 0));",
    "glob_inc": "{ int r1 = glob_inc(); int r2 = (int)DC(glob_inc)(0, 0, 0, 0, 0, 0, 0); CHK(\"glob_inc\", r1 + 1, r2); }",
    "call2":    "CHK(\"call2\", call2(k), (int)DC(call2)(k, 0, 0, 0, 0, 0, 0));",
    "sp":       "{ struct pt q = { k, n, L[3] }; CHK(\"sp\", sp(&q), DC(sp)((long)&q, 0, 0, 0, 0, 0, 0)); }",
    "fact":     "CHK(\"fact\", fact(n % 13), (int)DC(fact)(n % 13, 0, 0, 0, 0, 0, 0));",
    "sw":       "CHK(\"sw\", sw(m), (int)DC(sw)(m, 0, 0, 0, 0, 0, 0));",
    "fma3":     "{ double x = A[1] / 3.0, y = A[2] / 7.0, z = A[3] / 9.0; double r1 = fma3(x, y, z),"
                " r2 = ((double (*)(double, double, double))(void *)fma3_dc)(x, y, z); CHK(\"fma3\", memcmp(&r1, &r2, 8), 0); }",
    "copy":     "{ memset(D1, 1, 80); memset(D2, 1, 80); copy(D1, S); DC(copy)((long)D2, (long)S, 0, 0, 0, 0, 0);"
                " CHK(\"copy\", memcmp(D1, D2, 80), 0); }",
    "strlen2":  "CHK(\"strlen2\", strlen2(S), (int)DC(strlen2)((long)S, 0, 0, 0, 0, 0, 0));",
    "bits":     "CHK(\"bits\", bits((unsigned)L[4]), (unsigned)DC(bits)((unsigned)L[4], 0, 0, 0, 0, 0, 0));",
}
T2_FUNCS = ["local_arr", "use_addr", "fptr", "dense_sw", "udiv", "sdiv", "sh", "fcmp", "fl", "d2i", "nested",
            "many", "callmany", "whilebrk", "ll", "andor"]
T2_CASES = {
    "local_arr": "CHK(\"local_arr\", local_arr(k), (int)DC(local_arr)(k, 0, 0, 0, 0, 0, 0));",
    "use_addr":  "CHK(\"use_addr\", use_addr(k), (int)DC(use_addr)(k, 0, 0, 0, 0, 0, 0));",
    "fptr":      "CHK(\"fptr\", fptr(ext, k), (int)DC(fptr)((long)ext, k, 0, 0, 0, 0, 0));",
    "dense_sw":  "CHK(\"dense_sw\", dense_sw(m), (int)DC(dense_sw)(m, 0, 0, 0, 0, 0, 0));",
    "udiv":      "{ unsigned a = (unsigned)L[5], b = (unsigned)rnd(1, 1000); CHK(\"udiv\", udiv(a, b), (unsigned)DC(udiv)(a, b, 0, 0, 0, 0, 0)); }",
    "sdiv":      "{ int a = (int)L[6] * 997, b = (int)rnd(1, 50) * (n & 1 ? -1 : 1); CHK(\"sdiv\", sdiv(a, b), (int)DC(sdiv)(a, b, 0, 0, 0, 0, 0)); }",
    "sh":        "{ short a = (short)L[7]; unsigned char b = (unsigned char)k; CHK(\"sh\", sh(a, b), (short)DC(sh)(a, b, 0, 0, 0, 0, 0)); }",
    "fcmp":      "{ double x = A[1] / 4.0, y = A[2] / 4.0; CHK(\"fcmp\", fcmp(x, y), ((int (*)(double, double))(void *)fcmp_dc)(x, y)); }",
    "fl":        "{ float x = A[1] / 3.0f; float r1 = fl(x, k), r2 = ((float (*)(float, int))(void *)fl_dc)(x, k);"
                 " CHK(\"fl\", memcmp(&r1, &r2, 4), 0); }",
    "d2i":       "{ double x = L[8] / 7.0; CHK(\"d2i\", d2i(x), ((int (*)(double))(void *)d2i_dc)(x)); }",
    "nested":    "CHK(\"nested\", nested(n % 20, m), (int)DC(nested)(n % 20, m, 0, 0, 0, 0, 0));",
    "many":      "CHK(\"many\", many(A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8], A[9]),"
                 " ((int (*)(int, int, int, int, int, int, int, int, int, int))(void *)many_dc)(A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8], A[9]));",
    "callmany":  "CHK(\"callmany\", callmany(), (int)DC(callmany)(0, 0, 0, 0, 0, 0, 0));",
    "whilebrk":  "{ int P[80]; for (int i = 0; i < 80; i++) P[i] = (int)rnd(1, 9); P[rnd(0, 60)] = (int)rnd(-1, 0);"
                 " CHK(\"whilebrk\", whilebrk(P), (int)DC(whilebrk)((long)P, 0, 0, 0, 0, 0, 0)); }",
    "ll":        "CHK(\"ll\", ll(L[9]), DC(ll)(L[9], 0, 0, 0, 0, 0, 0));",
    "andor":     "CHK(\"andor\", andor(k, n, m), (int)DC(andor)(k, n, m, 0, 0, 0, 0));",
}
STRUCTS_FUNCS = ["pt_get", "pt_set", "mix_sum", "list_sum", "list_len", "buf_sum", "rec_find", "rec_total",
                 "rec_swap", "outer_get", "pt_copy", "pt_eq"]
STRUCTS_CASES = {
    "pt_get":    "{ struct pt p = {A[1], A[2], L[3]}; CHK(\"pt_get\", pt_get(&p), DC(pt_get)((long)&p, 0, 0, 0, 0, 0, 0)); }",
    "pt_set":    "{ struct pt p1 = {0, 0, 0}, p2 = {0, 0, 0}; pt_set(&p1, A[1], A[2]); DC(pt_set)((long)&p2, A[1], A[2], 0, 0, 0, 0);"
                 " CHK(\"pt_set\", memcmp(&p1, &p2, sizeof p1), 0); }",
    "mix_sum":   "{ struct mix x = {(char)A[1], (short)(A[2] * 100), A[3] * 1000, L[4] / 3.0, L[5] * 100000};"
                 " double r1 = mix_sum(&x), r2 = ((double (*)(const struct mix *))(void *)mix_sum_dc)(&x);"
                 " CHK(\"mix_sum\", memcmp(&r1, &r2, 8), 0); }",
    "list_sum":  "{ struct node N[10]; int len = (int)rnd(0, 10); for (int i = 0; i < len; i++) { N[i].val = A[i];"
                 " N[i].next = i + 1 < len ? &N[i + 1] : 0; } struct node *h = len ? N : 0;"
                 " CHK(\"list_sum\", list_sum(h), (int)DC(list_sum)((long)h, 0, 0, 0, 0, 0, 0)); }",
    "list_len":  "{ struct node N[10]; int len = (int)rnd(0, 10); for (int i = 0; i < len; i++) { N[i].val = A[i];"
                 " N[i].next = i + 1 < len ? &N[i + 1] : 0; } struct node *h = len ? N : 0;"
                 " CHK(\"list_len\", list_len(h), (int)DC(list_len)((long)h, 0, 0, 0, 0, 0, 0)); }",
    "buf_sum":   "{ struct buf b; b.len = (int)rnd(-2, 20); for (int i = 0; i < 20; i++) b.data[i] = (char)A[i];"
                 " CHK(\"buf_sum\", buf_sum(&b), (int)DC(buf_sum)((long)&b, 0, 0, 0, 0, 0, 0)); }",
    "rec_find":  "{ struct rec r[40]; for (int i = 0; i < 40; i++) { r[i].key = A[i] % 9; r[i].val = A[i + 1]; }"
                 " int nn = n % 41, key = A[rnd(0, 50)] % 9;"
                 " CHK(\"rec_find\", rec_find(r, nn, key), (int)DC(rec_find)((long)r, nn, key, 0, 0, 0, 0)); }",
    "rec_total": "{ struct rec r[40]; for (int i = 0; i < 40; i++) { r[i].key = A[i]; r[i].val = A[i + 2] * 1000; }"
                 " int nn = n % 41; CHK(\"rec_total\", rec_total(r, nn), DC(rec_total)((long)r, nn, 0, 0, 0, 0, 0)); }",
    "rec_swap":  "{ struct rec r1[40], r2[40]; for (int i = 0; i < 40; i++) { r1[i].key = r2[i].key = A[i];"
                 " r1[i].val = r2[i].val = A[i + 3]; } int nn = n % 41; rec_swap(r1, nn);"
                 " DC(rec_swap)((long)r2, nn, 0, 0, 0, 0, 0); CHK(\"rec_swap\", memcmp(r1, r2, sizeof r1), 0); }",
    "outer_get": "{ struct outer o; memset(&o, 0, sizeof o); o.tag = A[1]; o.in.a = (short)A[2]; o.in.b = (short)A[3];"
                 " o.sum = L[4]; CHK(\"outer_get\", outer_get(&o), (int)DC(outer_get)((long)&o, 0, 0, 0, 0, 0, 0)); }",
    "pt_copy":   "{ struct pt s = {A[1], A[2], L[3]}, d1 = {0, 0, 0}, d2 = {0, 0, 0}; pt_copy(&d1, &s);"
                 " DC(pt_copy)((long)&d2, (long)&s, 0, 0, 0, 0, 0); CHK(\"pt_copy\", memcmp(&d1, &d2, sizeof d1), 0); }",
    "pt_eq":     "{ struct pt a = {A[1] % 3, A[2] % 3, L[3] % 3}, b = {A[4] % 3, A[5] % 3, L[6] % 3}; if (n & 1) b = a;"
                 " CHK(\"pt_eq\", pt_eq(&a, &b), (int)DC(pt_eq)((long)&a, (long)&b, 0, 0, 0, 0, 0)); }",
}

STRUCTS2_FUNCS = ["first_chars", "push", "mat_sum", "inner_sum", "outer_total", "inner2_sum", "outer2_total",
                  "buf_clear", "buf_last",
                  "pt_init", "pt_local", "gpt_get", "gpt_set", "gtab_find"]
STRUCTS2_CASES = {
    "first_chars": "{ char S8[8][4]; char *V[8]; for (int i = 0; i < 8; i++) { S8[i][0] = (char)A[i]; V[i] = S8[i]; }"
                   " int nn = (int)rnd(0, 8); CHK(\"first_chars\", first_chars(V, nn), (int)DC(first_chars)((long)V, nn, 0, 0, 0, 0, 0)); }",
    "push":        "{ struct node o1, o2, n1, n2; struct node *h1 = &o1, *h2 = &o2; push(&h1, &n1); DC(push)((long)&h2, (long)&n2, 0, 0, 0, 0, 0);"
                   " CHK(\"push\", h1 == &n1 && n1.next == &o1, h2 == &n2 && n2.next == &o2); }",
    "mat_sum":     "{ int M[5][7]; int *R[5]; for (int i = 0; i < 5; i++) { R[i] = M[i]; for (int j = 0; j < 7; j++) M[i][j] = A[i * 7 + j]; }"
                   " int r = (int)rnd(0, 5), c = (int)rnd(0, 7); CHK(\"mat_sum\", mat_sum(R, r, c), DC(mat_sum)((long)R, r, c, 0, 0, 0, 0)); }",
    "inner_sum":   "{ struct inner in = {(short)A[1], (short)A[2]}; CHK(\"inner_sum\", inner_sum(&in), (int)DC(inner_sum)((long)&in, 0, 0, 0, 0, 0, 0)); }",
    "outer_total": "{ struct outer o; memset(&o, 0, sizeof o); o.tag = A[1]; o.in.a = (short)A[2]; o.in.b = (short)A[3]; o.sum = L[4];"
                   " CHK(\"outer_total\", outer_total(&o), (int)DC(outer_total)((long)&o, 0, 0, 0, 0, 0, 0)); }",
    "inner2_sum":  "{ struct inner2 in = {(short)A[1], A[2] * 1000}; CHK(\"inner2_sum\", inner2_sum(&in), (int)DC(inner2_sum)((long)&in, 0, 0, 0, 0, 0, 0)); }",
    "outer2_total": "{ struct outer2 o; memset(&o, 0, sizeof o); o.tag = A[1]; o.in.a = (short)A[2]; o.in.b = A[3] * 1000; o.sum = L[4];"
                    " CHK(\"outer2_total\", outer2_total(&o), (int)DC(outer2_total)((long)&o, 0, 0, 0, 0, 0, 0)); }",
    "buf_clear":   "{ struct buf b1, b2; memset(&b1, 0x55, sizeof b1); memset(&b2, 0x55, sizeof b2); buf_clear(&b1);"
                   " DC(buf_clear)((long)&b2, 0, 0, 0, 0, 0, 0); CHK(\"buf_clear\", memcmp(&b1, &b2, sizeof b1), 0); }",
    "buf_last":    "{ struct buf b; b.len = A[2]; for (int i = 0; i < 20; i++) b.data[i] = (char)A[i + 3];"
                   " CHK(\"buf_last\", buf_last(&b), (int)DC(buf_last)((long)&b, 0, 0, 0, 0, 0, 0)); }",
    "pt_init":     "{ struct pt p1, p2; memset(&p1, 0, sizeof p1); memset(&p2, 0, sizeof p2); pt_init(&p1, A[1], A[2]);"
                   " DC(pt_init)((long)&p2, A[1], A[2], 0, 0, 0, 0); CHK(\"pt_init\", memcmp(&p1, &p2, sizeof p1), 0); }",
    "pt_local":    "CHK(\"pt_local\", pt_local(A[1], A[2]), DC(pt_local)(A[1], A[2], 0, 0, 0, 0, 0));",
    "gpt_get":     "{ gpt.x = A[1]; gpt.y = A[2]; gpt.z = L[3]; CHK(\"gpt_get\", gpt_get(), DC(gpt_get)(0, 0, 0, 0, 0, 0, 0)); }",
    "gpt_set":     "{ gpt_set(A[1], A[2]); struct pt save = gpt; memset(&gpt, 0, sizeof gpt); DC(gpt_set)(A[1], A[2], 0, 0, 0, 0, 0);"
                   " CHK(\"gpt_set\", memcmp(&save, &gpt, sizeof gpt), 0); }",
    "gtab_find":   "{ for (int i = 0; i < 16; i++) { gtab[i].key = A[i] % 9; gtab[i].val = A[i + 1]; } int key = A[rnd(0, 40)] % 9;"
                   " CHK(\"gtab_find\", gtab_find(key), (int)DC(gtab_find)(key, 0, 0, 0, 0, 0, 0)); }",
}

# tests/loops/unroll.c: loops lcc unrolls (reroll.py).  dsum is left out: at -O2 its
# pointer parameter shares %dr0 with the double result (predicated writes join the
# two), so the decompiled prototype says `double a1`.
UNROLL_FUNCS = ["iota", "dot2", "sum_down", "count_pos", "sum_short", "dec_chars", "axpy", "sum_even",
                "sum_from", "sum_xor", "sum2d", "widen"]
UNROLL_CASES = {
    "iota":      "{ int X[80], Y[80]; memset(X, 0, sizeof X); memset(Y, 0, sizeof Y); iota(X, n);"
                 " DC(iota)((long)Y, n, 0, 0, 0, 0, 0); CHK(\"iota\", memcmp(X, Y, sizeof X), 0); }",
    "dot2":      "CHK(\"dot2\", dot2(A, B, n), DC(dot2)((long)A, (long)B, n, 0, 0, 0, 0));",
    "sum_down":  "CHK(\"sum_down\", sum_down(A, n), (int)DC(sum_down)((long)A, n, 0, 0, 0, 0, 0));",
    "count_pos": "CHK(\"count_pos\", count_pos(A, n), (int)DC(count_pos)((long)A, n, 0, 0, 0, 0, 0));",
    "sum_short": "{ short H[80]; for (int i = 0; i < 80; i++) H[i] = (short)(A[i] * 600);"
                 " CHK(\"sum_short\", sum_short(H, n), (int)DC(sum_short)((long)H, n, 0, 0, 0, 0, 0)); }",
    "dec_chars": "{ memcpy(D1, S, 80); memcpy(D2, S, 80); dec_chars(D1, n); DC(dec_chars)((long)D2, n, 0, 0, 0, 0, 0);"
                 " CHK(\"dec_chars\", memcmp(D1, D2, 80), 0); }",
    "axpy":      "{ int X[80], Y[80]; for (int i = 0; i < 80; i++) X[i] = Y[i] = (int)rnd(-99, 99); axpy(X, A, k, n);"
                 " DC(axpy)((long)Y, (long)A, k, n, 0, 0, 0); CHK(\"axpy\", memcmp(X, Y, sizeof X), 0);"
                 " int d = (int)rnd(1, 3), m2 = n < 70 ? n : 70; for (int i = 0; i < 80; i++) X[i] = Y[i] = (int)rnd(-9, 9);"
                 " axpy(X + d, X, k, m2); DC(axpy)((long)(Y + d), (long)Y, k, m2, 0, 0, 0);"
                 " CHK(\"axpy_overlap\", memcmp(X, Y, sizeof X), 0); }",
    "sum_even":  "CHK(\"sum_even\", sum_even(A, n), (int)DC(sum_even)((long)A, n, 0, 0, 0, 0, 0));",
    "sum_from":  "{ int k0 = (int)rnd(0, 30); CHK(\"sum_from\", sum_from(A, k0, n), (int)DC(sum_from)((long)A, k0, n, 0, 0, 0, 0)); }",
    "sum_xor":   "{ int x1 = 5, x2 = 5; int r1 = sum_xor(A, n, &x1), r2 = (int)DC(sum_xor)((long)A, n, (long)&x2, 0, 0, 0, 0);"
                 " CHK(\"sum_xor\", r1, r2); CHK(\"sum_xor_out\", x1, x2); }",
    "sum2d":     "{ int r = (int)rnd(-1, 8), c = (int)rnd(-1, 9); CHK(\"sum2d\", sum2d(A, r, c), DC(sum2d)((long)A, r, c, 0, 0, 0, 0)); }",
    "widen":     "{ long X[80], Y[80]; memset(X, 0, sizeof X); memset(Y, 0, sizeof Y); widen(X, A, n);"
                 " DC(widen)((long)Y, (long)A, n, 0, 0, 0, 0); CHK(\"widen\", memcmp(X, Y, sizeof X), 0); }",
}

PRELUDE = """
static inline float __bitcast_float(int x) { float f; __builtin_memcpy(&f, &x, 4); return f; }
static inline int __bitcast_int(float f) { int x; __builtin_memcpy(&x, &f, 4); return x; }
static inline double __bitcast_double(long x) { double f; __builtin_memcpy(&f, &x, 8); return f; }
static inline long __bitcast_long(double f) { long x; __builtin_memcpy(&x, &f, 8); return x; }
"""


def function_blocks(code):
    """name -> text of each function definition in a decompiled file"""
    out = {}
    for m in re.finditer(r"^(?:// [^\n]*\n)*[A-Za-z_][^\n;]*?\b(\w+)\([^\n;]*\)\n\{\n.*?^\}\n?", code, re.M | re.S):
        out[m.group(1)] = m.group(0)
    return out


def run(obj):
    base = os.path.basename(obj)
    dc_prelude = ""
    if base.startswith("arch"):
        funcs, cases, src = ARCH_FUNCS, ARCH_CASES, os.path.join(ROOT, "tests", "arch", "arch.c")
    elif base.startswith("t1"):
        funcs, cases, src = T1_FUNCS, T1_CASES, os.path.join(ROOT, "tests", "t1.c")
        dc_prelude = "extern int g;\n"
    elif base.startswith("structs2"):
        funcs, cases, src = STRUCTS2_FUNCS, STRUCTS2_CASES, os.path.join(ROOT, "tests", "structs", "structs2.c")
    elif base.startswith("structs"):
        funcs, cases, src = STRUCTS_FUNCS, STRUCTS_CASES, os.path.join(ROOT, "tests", "structs", "structs.c")
    elif base.startswith("unroll"):
        funcs, cases, src = UNROLL_FUNCS, UNROLL_CASES, os.path.join(HERE, "unroll.c")
    elif base.startswith("t2"):
        funcs, cases, src = T2_FUNCS, T2_CASES, os.path.join(ROOT, "tests", "t2.c")
        dc_prelude = "void take(int *);\n"
    else:
        funcs, cases, src = FUNCS, CASES, os.path.join(HERE, "loops.c")
    d = Decompiler(obj, objdump=os.path.join(ROOT, "tests", "e2k-linux-objdump"))
    # the whole file at once: types are inferred across functions (struct of a
    # callee's parameter, globals), as in a normal run
    full = d.decompile(header=True)
    blocks = function_blocks(full)
    first = min((full.index(b) for b in blocks.values()), default=len(full))
    head = full[:first]
    head = re.sub(r"\b(%s)\(" % "|".join(funcs), lambda m: m.group(1) + "_dc(", head)
    # the harness declares the external helpers itself
    head = re.sub(r"^[^\n]*\b(ext|take)\([^\n]*\);\n", "", head, flags=re.M)
    bodies, low, tested = [], [], []
    for name in funcs:
        code = blocks.get(name, "")
        if not code.strip() or "decompilation failed" in code:
            low.append(name + "(failed)")
            continue
        if "__e2k_" in code or "__asm__" in code:
            low.append(name)
            continue
        code = re.sub(r"\b(%s)\(" % "|".join(funcs), lambda m: m.group(1) + "_dc(", code)
        bodies.append(code)
        tested.append(name)
    protos = "\n".join("long %s_dc();" % n for n in tested)
    # the original's own main() would clash with the harness
    origs = "#define main orig_main\n#include \"%s\"\n#undef main\n" % src
    calls = "\n".join("    " + cases[n] for n in tested)
    with tempfile.TemporaryDirectory() as td:
        dcf = os.path.join(td, "dc.c")
        with open(dcf, "w") as f:
            f.write("int ext(int);\n" + dc_prelude)
            f.write(head)
            f.write("\n\n".join(bodies))
        hf = os.path.join(td, "h.c")
        with open(hf, "w") as f:
            f.write(origs)
            f.write(HARNESS % {"protos": protos, "calls": calls})
        exe = os.path.join(td, "t")
        flags = ["-O0", "-w", "-fwrapv", "-fno-strict-aliasing", "-Wno-int-conversion",
                 "-Wno-incompatible-pointer-types", "-Wno-implicit-function-declaration"]
        r = subprocess.run(["gcc"] + flags + [hf, dcf, "-o", exe], capture_output=True, text=True)
        if r.returncode:
            print("compile error:\n" + r.stderr[-3000:])
            return False
        print("%s: tested %d (%s)" % (os.path.basename(obj), len(tested), " ".join(tested)))
        if low:
            print("   low-level, not tested: " + " ".join(low))
        try:
            r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            print("   TIMEOUT (a decompiled loop does not terminate)")
            return False
        print("   " + r.stdout.strip().replace("\n", "\n   "))
        if r.returncode < 0:
            print("   CRASHED (signal %d)" % -r.returncode)
        return r.returncode == 0


if __name__ == "__main__":
    objs = sys.argv[1:] or [os.path.join(HERE, "loops_O%d.o" % i) for i in range(4)] + \
        [os.path.join(HERE, "unroll_O%d.o" % i) for i in (1, 2, 3)] + \
        [os.path.join(ROOT, "tests", "arch", "arch_O%d.o" % i) for i in (1, 3)] + \
        [os.path.join(ROOT, "tests", "t%d_O%d.o" % (t, i)) for t in (1, 2) for i in range(3)] + \
        [os.path.join(ROOT, "tests", "structs", "%s_O%d.o" % (b, i)) for b in ("structs", "structs2") for i in range(4)]
    ok = all([run(o) for o in objs])
    sys.exit(0 if ok else 1)
