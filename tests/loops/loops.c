/* Тесты на восстановление циклов for / while / do-while.
 * Сборка:  lcc -O{0,1,2,3} -c loops.c -o loops_O{N}.o
 * Каждая функция — отдельная форма цикла, которую декомпилятор должен
 * вернуть в читаемом виде (for / while / do-while, break / continue). */
int ext(int);

/* 1. Классический for со счётчиком вверх. */
int for_up(const int *p, int n)
{
    int s = 0;
    for (int i = 0; i < n; i++)
        s += p[i];
    return s;
}

/* 2. Счётчик вниз до нуля включительно. */
int for_down(const int *p, int n)
{
    int s = 0;
    for (int i = n - 1; i >= 0; i--)
        s = s * 3 + p[i];
    return s;
}

/* 3. Шаг не единица. */
int for_step(const int *p, int n)
{
    int s = 0;
    for (int i = 0; i < n; i += 3)
        s ^= p[i];
    return s;
}

/* 4. Константное число итераций. */
int for_const(const int *p)
{
    int s = 0;
    for (int i = 0; i < 10; i++)
        s += p[i] * i;
    return s;
}

/* 5. Итерация указателем до конца. */
long for_ptr(const long *p, const long *end)
{
    long s = 0;
    for (; p != end; p++)
        s += *p;
    return s;
}

/* 6. while по указателю (strlen). */
int while_ptr(const char *s)
{
    const char *b = s;
    while (*s)
        s++;
    return s - b;
}

/* 7. while с нетривиальным телом (Коллатц). */
int while_cond(unsigned x)
{
    int n = 0;
    while (x > 1) {
        if (x & 1)
            x = 3 * x + 1;
        else
            x >>= 1;
        n++;
    }
    return n;
}

/* 8. do-while: тело выполняется хотя бы раз. */
int do_while(int n)
{
    int s = 0;
    do {
        s += n;
        n -= 2;
    } while (n > 0);
    return s;
}

/* 9. Вложенные for. */
int nested2(int n, int m)
{
    int s = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            s += i * j;
    return s;
}

/* 10. for с break. */
int for_break(const int *p, int n, int k)
{
    int i;
    for (i = 0; i < n; i++)
        if (p[i] == k)
            break;
    return i;
}

/* 11. for с continue. */
int for_continue(const int *p, int n)
{
    int s = 0;
    for (int i = 0; i < n; i++) {
        if (p[i] < 0)
            continue;
        s += p[i];
    }
    return s;
}

/* 12. Цикл с вызовом внутри (мешает конвейеризации). */
int while_call(int n)
{
    int s = 0;
    while (n > 0) {
        s += ext(n);
        n--;
    }
    return s;
}

/* 13. Бесконечный цикл с выходом из середины. */
int inf_loop(const int *p)
{
    int i = 0;
    for (;;) {
        int v = p[i];
        if (v == 0)
            break;
        i += v;
    }
    return i;
}

/* 14. Цикл записи (memset). */
void fill(char *d, char c, int n)
{
    for (int i = 0; i < n; i++)
        d[i] = c;
}

/* 15. Беззнаковый счётчик, условие !=. */
unsigned for_unsigned(unsigned n)
{
    unsigned s = 0;
    for (unsigned i = 0; i != n; i++)
        s += i * i;
    return s;
}

/* 16. Копирование массива (load + store). */
void copy_arr(int *d, const int *s, int n)
{
    for (int i = 0; i < n; i++)
        d[i] = s[i] + 1;
}
