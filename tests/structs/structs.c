/* Тесты распознавания структур: поля разных типов, указатель на ту же
   структуру (список), массив внутри структуры, массив структур,
   вложенная структура, запись полей, два указателя одного типа. */

struct pt { int x, y; long z; };
struct mix { char c; short s; int i; double d; long l; };
struct node { int val; struct node *next; };
struct buf { int len; char data[20]; };
struct rec { int key; int val; };
struct inner { short a; short b; };
struct outer { int tag; struct inner in; long sum; };

/* 1. Чтение полей разных типов. */
long pt_get(struct pt *p)
{
    return p->x * p->y + p->z;
}

/* 2. Запись полей. */
void pt_set(struct pt *p, int x, int y)
{
    p->x = x;
    p->y = y;
    p->z = (long)x * y;
}

/* 3. char, short, int, double, long с выравниванием. */
double mix_sum(const struct mix *m)
{
    return m->c + m->s + m->i + m->d + m->l;
}

/* 4. Список: поле-указатель на ту же структуру. */
int list_sum(const struct node *n)
{
    int s = 0;
    while (n) {
        s += n->val;
        n = n->next;
    }
    return s;
}

/* 5. То же, for с подсчётом. */
int list_len(const struct node *n)
{
    int k = 0;
    for (; n; n = n->next)
        k++;
    return k;
}

/* 6. Массив внутри структуры. */
int buf_sum(const struct buf *b)
{
    int s = 0;
    for (int i = 0; i < b->len; i++)
        s += b->data[i];
    return s;
}

/* 7. Массив структур: поиск. */
int rec_find(const struct rec *r, int n, int key)
{
    for (int i = 0; i < n; i++)
        if (r[i].key == key)
            return r[i].val;
    return -1;
}

/* 8. Массив структур: сумма произведений полей. */
long rec_total(const struct rec *r, int n)
{
    long s = 0;
    for (int i = 0; i < n; i++)
        s += (long)r[i].key * r[i].val;
    return s;
}

/* 9. Массив структур: запись. */
void rec_swap(struct rec *r, int n)
{
    for (int i = 0; i < n; i++) {
        int t = r[i].key;
        r[i].key = r[i].val;
        r[i].val = t;
    }
}

/* 10. Вложенная структура. */
int outer_get(const struct outer *o)
{
    return o->tag + o->in.a * o->in.b + (int)o->sum;
}

/* 11. Копирование между двумя указателями одного типа. */
void pt_copy(struct pt *d, const struct pt *s)
{
    d->x = s->x;
    d->y = s->y;
    d->z = s->z;
}

/* 12. Сравнение двух структур. */
int pt_eq(const struct pt *a, const struct pt *b)
{
    return a->x == b->x && a->y == b->y && a->z == b->z;
}
