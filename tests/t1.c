int add(int a, int b) { return a + b; }
long mul3(long a, long b, long c) { return a * b + c; }
int max(int a, int b) { return a > b ? a : b; }
int absdiff(int a, int b) { if (a > b) return a - b; else return b - a; }
int sum(int *p, int n) { int s = 0; for (int i = 0; i < n; i++) s += p[i]; return s; }
int g;
int glob_inc(void) { return ++g; }
int ext(int);
int call2(int x) { return ext(x) + ext(x + 1); }
struct pt { int x, y; long z; };
long sp(struct pt *p) { return p->x * p->y + p->z; }
int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }
int sw(int x) { switch (x) { case 0: return 10; case 1: return 20; case 2: return 33; case 3: return 47; case 5: return 9; default: return -1; } }
double fma3(double a, double b, double c) { return a * b + c; }
void copy(char *d, const char *s) { while ((*d++ = *s++)); }
int strlen2(const char *s) { int n = 0; while (s[n]) n++; return n; }
unsigned bits(unsigned x) { return (x >> 3) ^ (x << 5) | (x & 0xff); }
