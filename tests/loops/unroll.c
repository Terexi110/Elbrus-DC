/* Циклы, которые lcc разворачивает: на -O2 тело повторено дважды, а при
 * нечётном числе итераций после цикла выполняется ещё одна копия.
 * Сборка:  lcc -O{1,2,3} -c unroll.c -o unroll_O{N}.o
 * Декомпилятор должен вернуть один цикл с одной копией тела (reroll.py). */

/* 1. Индекс участвует в значении. */
void iota(int *d, int n)
{
    for (int i = 0; i < n; i++)
        d[i] = i * 3 + 1;
}

/* 2. Скалярное произведение, накопление в long. */
long dot2(const int *a, const int *b, int n)
{
    long s = 0;
    for (int i = 0; i < n; i++)
        s += (long)a[i] * b[i];
    return s;
}

/* 3. Счётчик вниз. */
int sum_down(const int *a, int n)
{
    int s = 0;
    for (int i = n - 1; i >= 0; i--)
        s = s * 31 + a[i];
    return s;
}

/* 4. Условие в теле. */
int count_pos(const int *a, int n)
{
    int c = 0;
    for (int i = 0; i < n; i++)
        if (a[i] > 0)
            c++;
    return c;
}

/* 5. short и char. */
int sum_short(const short *a, int n)
{
    int s = 0;
    for (int i = 0; i < n; i++)
        s += a[i];
    return s;
}

void dec_chars(char *s, int n)
{
    for (int i = 0; i < n; i++)
        s[i] = s[i] - 32;
}

/* 6. Изменение на месте с инвариантом. */
void axpy(int *y, const int *x, int a, int n)
{
    for (int i = 0; i < n; i++)
        y[i] += a * x[i];
}

/* 7. Шаг 2. */
int sum_even(const int *a, int n)
{
    int s = 0;
    for (int i = 0; i < n; i += 2)
        s += a[i];
    return s;
}

/* 8. Начало не с нуля. */
int sum_from(const int *a, int k, int n)
{
    int s = 0;
    for (int i = k; i < n; i++)
        s += a[i];
    return s;
}

/* 9. Два накопителя, результат через указатель. */
int sum_xor(const int *a, int n, int *x)
{
    int s = 0, t = 0;
    for (int i = 0; i < n; i++) {
        s += a[i];
        t ^= a[i] << 1;
    }
    *x = t;
    return s;
}

/* 10. Вложенные циклы, развёрнут внутренний. */
long sum2d(const int *a, int r, int c)
{
    long s = 0;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            s += a[i * c + j];
    return s;
}

/* 11. Расширение int -> long. */
void widen(long *d, const int *s, int n)
{
    for (int i = 0; i < n; i++)
        d[i] = s[i];
}

/* 12. Плавающая точка: порядок сложений сохраняется. */
double dsum(const double *a, int n)
{
    double s = 0;
    for (int i = 0; i < n; i++)
        s += a[i];
    return s;
}
