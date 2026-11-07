/* Тесты второй очереди: указатели на указатели, вложенная структура через
   указатель на поле, длина массива в структуре, структура на стеке,
   глобальная структура и глобальный массив структур. */

struct pt { int x, y; long z; };
struct node { int val; struct node *next; };
struct inner { short a; short b; };
struct outer { int tag; struct inner in; long sum; };
struct buf { int len; char data[20]; };
struct rec { int key; int val; };

#define NOINLINE __attribute__((noinline))

/* 1. char **: первый символ каждой строки. */
int first_chars(char **v, int n)
{
    int s = 0;
    for (int i = 0; i < n; i++)
        s += v[i][0];
    return s;
}

/* 2. struct node **: вставка в начало списка. */
void push(struct node **head, struct node *n)
{
    n->next = *head;
    *head = n;
}

/* 3. int **: сумма матрицы, заданной массивом строк. */
long mat_sum(int **m, int rows, int cols)
{
    long s = 0;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            s += m[i][j];
    return s;
}

/* 4. Вложенная структура: указатель на поле передаётся в функцию. */
NOINLINE int inner_sum(const struct inner *p)
{
    return p->a * 3 + p->b;
}

int outer_total(const struct outer *o)
{
    return o->tag + inner_sum(&o->in) + (int)o->sum;
}

/* 4b. То же со вложенной структурой из полей разного размера. */
struct inner2 { short a; int b; };
struct outer2 { int tag; struct inner2 in; long sum; };

NOINLINE int inner2_sum(const struct inner2 *p)
{
    return p->a * 3 + p->b;
}

int outer2_total(const struct outer2 *o)
{
    return o->tag + inner2_sum(&o->in) + (int)o->sum;
}

/* 5. Длина массива внутри структуры: константный цикл и последний элемент. */
void buf_clear(struct buf *b)
{
    b->len = 0;
    for (int i = 0; i < 20; i++)
        b->data[i] = (char)i;
}

int buf_last(const struct buf *b)
{
    return b->data[19] + b->len;
}

/* 6. Структура на стеке, её адрес передаётся в функцию. */
NOINLINE void pt_init(struct pt *p, int x, int y)
{
    p->x = x;
    p->y = y;
    p->z = (long)x * y;
}

long pt_local(int x, int y)
{
    struct pt p;
    pt_init(&p, x, y);
    return p.z + p.x - p.y;
}

/* 7. Глобальная структура. */
struct pt gpt;

long gpt_get(void)
{
    return gpt.x * gpt.y + gpt.z;
}

void gpt_set(int x, int y)
{
    gpt.x = x;
    gpt.y = y;
    gpt.z = (long)x - y;
}

/* 8. Глобальный массив структур. */
struct rec gtab[16];

int gtab_find(int key)
{
    for (int i = 0; i < 16; i++)
        if (gtab[i].key == key)
            return gtab[i].val;
    return -1;
}
