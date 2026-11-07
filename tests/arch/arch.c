/* Tests that expose Elbrus (e2k) architecture features.
 * Each function is written so that lcc -O3 has a reason to use one feature.
 * See README.md next to this file for what to look for in the disassembly. */

/* 1. "Две столовые": обе ветки считаются одновременно, результат выбирается
 *    предикатом (if-conversion). Ожидается: cmp* -> %pred, операции с "? %pred",
 *    отсутствие ct/disp внутри функции, кроме return. */
int canteen(int open1, int a, int b)
{
    int r;
    if (open1)
        r = a * 3 + 7;      /* путь в первую столовую */
    else
        r = b * 5 - 2;      /* путь во вторую */
    return r;
}

/* 2. То же, но с вложенными условиями: несколько предикатов, их комбинации
 *    в PLU (pass / landp / andp). */
int canteen3(int x, int y, int a, int b, int c)
{
    if (x > 0 && y > 0)
        return a + b;
    if (x < 0 || y == 7)
        return b - c;
    return a ^ c;
}

/* 3. Спекулятивная загрузка: чтение памяти до проверки указателя.
 *    Ожидается: ld* с флагом ",sm" (speculative mode) раньше проверки. */
int spec_load(int *p, int def)
{
    if (p != 0)
        return *p + 1;
    return def;
}

/* 4. Широкая команда: много независимых операций, которые компилятор
 *    упакует в одну-две ШК по разным каналам ALU (,0 ... ,5). */
long wide(long a, long b, long c, long d, long e, long f)
{
    long x0 = a + b, x1 = c - d, x2 = e & f;
    long x3 = a ^ c, x4 = b | d, x5 = e + 13;
    return (x0 * x1) + (x2 * x3) + (x4 * x5);
}

/* 5. Программно-конвейеризованный цикл: вращающиеся регистры %b[] и
 *    счётчик цикла. Ожидается: setbn/abn/alc, loop_mode, ct ? %NOT_LOOP_END. */
void saxpy(int n, float a, const float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}

/* 6. Устройство асинхронной подкачки массивов (APB/AAU): потоковое чтение
 *    массива. Ожидается: aau*, mova*, aaurw*, staa*, ldaa*. */
long dot(const int *a, const int *b, int n)
{
    long s = 0;
    for (int i = 0; i < n; i++)
        s += (long)a[i] * b[i];
    return s;
}

/* 7. Регистровые окна: глубокая рекурсия без сохранения регистров в стек
 *    вручную. Ожидается: setwd/setbn, call ... wbs = N, аргументы в %b[]. */
int fib(int n)
{
    return n < 2 ? n : fib(n - 1) + fib(n - 2);
}

/* 8. Многоальтернативный выбор без переходов: цепочка min/max/abs
 *    превращается в merge* по предикатам. */
int clamp_abs(int v, int lo, int hi)
{
    if (v < 0) v = -v;
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v;
}

/* 9. Ранний выход из цикла поиска: цикл с двумя выходами,
 *    предикаты условий выхода считаются наперёд. */
int find(const int *p, int n, int key)
{
    for (int i = 0; i < n; i++)
        if (p[i] == key)
            return i;
    return -1;
}

/* 10. Плавающая точка с выбором: обе ветки вычисляются в FP-каналах,
 *     затем выбирается результат. */
double fsel(double x, double y)
{
    return x > y ? x * 2.0 + y : y * 0.5 - x;
}
