#include <stdio.h>
#include <string.h>
int printf(const char*, ...);
int local_arr(int n) { int a[16]; for (int i = 0; i < 16; i++) a[i] = i * n; return a[n & 15]; }
int use_addr(int x) { int y = x; void take(int*); take(&y); return y; }
int fptr(int (*f)(int), int x) { return f(x) * 2; }
int dense_sw(int x) { switch (x) { case 0: return 3; case 1: return 7; case 2: return 1; case 3: return 9; case 4: return 12; case 5: return 4; case 6: return 8; case 7: return 2; } return 0; }
unsigned udiv(unsigned a, unsigned b) { return a / b + a % b; }
int sdiv(int a, int b) { return a / b; }
short sh(short a, unsigned char b) { return a * b; }
int fcmp(double a, double b) { return a < b ? 1 : (a == b ? 0 : -1); }
float fl(float a, int b) { return a * (float)b; }
int d2i(double d) { return (int)d; }
int nested(int n, int m) { int s = 0; for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) s += i ^ j; return s; }
void hello(const char *name, int n) { printf("hello %s %d\n", name, n); }
int many(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j) { return a+b+c+d+e+f+g+h+i+j; }
int callmany(void) { return many(1,2,3,4,5,6,7,8,9,10); }
int whilebrk(int *p) { int i = 0; while (1) { if (p[i] < 0) break; if (p[i] == 0) return -1; i++; } return i; }
long ll(long a) { return a > 100 ? a / 7 : a * 3; }
int andor(int a, int b, int c) { if (a > 0 && b > 0) return 1; if (a < 0 || c == 5) return 2; return 3; }
const char *str(int i) { static const char *t[] = {"a","bb","ccc"}; return t[i]; }
int main(int argc, char **argv) { hello(argv[0], argc); return nested(argc, 3); }
