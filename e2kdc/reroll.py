"""Re-rolling of unrolled counted loops.

At -O2 lcc unrolls a simple counted loop twice and finishes an odd count with
one more copy of the body after the loop:

    v3 = n > 0 ? n : 1;  v7 = (unsigned)v3 >> 1;          N = max(n, 1), M = N / 2
    t = 0;
    if ( v7 != 0 ) {
        v13 = v7 <= 0 ? 1 : v7;                             = M
        t = (long)v13 << 1;                                 = 2 * M
        for ( i = 0; i < v13; ++i )
            s = a[2 * i] + a[2 * i + 1] + s;                two iterations, fused
    }
    if ( v7 << 1 != v3 )                                    N odd
        s = s + a[t];                                       the last iteration

and this pass prints it back as

    for ( i = 0; i < n; ++i )
        s = s + a[i];

How it is proved, for an unroll factor k (2 here):

* the region is: plain assignments, the `if (M != 0)` holding the loop (or
  if / else, with plain assignments around the loop), plain assignments,
  the tail `if (TC) T` (also `if (!TC) return v; return E;`, i.e. T is
  `v = E`, and `if (!TC) return; T` at the end of a void function);
* the values of the variables the tail reads are computed on both paths
  (M == 0 and M != 0) by substitution; on the loop path `M <= 0 ? 1 : M`
  is M (the loop runs M times); the two paths must agree once M == 0 is
  put into the loop path;
* TC must be `k * M != N` with M = N >> 1 (unsigned): the tail runs exactly
  when N is odd, so 2 * M + (N & 1) = N iterations run in all;
* the tail, with those values put in, mentions M only as k * M * c, which
  is the index J = k * M of the last iteration: replacing k * M by J gives
  the body S(J) of one iteration;
* the loop body must do what S(k*i); S(k*i + 1) do: both are run
  symbolically (variables and stores) and the final values compared modulo
  associativity and commutativity of + ^ | & and linear arithmetic; loads
  read what an earlier store of the same iteration wrote to the same
  address and memory otherwise (the compiler already reordered loads and
  stores of the copies, so it proved they do not alias);
* N must be non-negative (it is max(n, 1) or the count proved positive).

Details of the region that are accepted:

* the loop alone, without `if (M != 0)`, running max(M, 1) times, when the
  facts give N >= 2 (the vectorised versions of -O3 are entered for n >= 16);
* `N & -2` for k * M, in the tail condition and in the tail (iota: d[t] = 3 * t + 1);
* a variable the body sets to a function of the index only (t = 2 * i): its
  value after the loop is that of the last iteration, i = M - 1;
* a load of the tail hoisted between the loop and the tail (`x = a[t]`, read
  only by the tail) goes with the tail.

Variables used only by the old region are dropped afterwards.

After re-rolling (and for loops that were never unrolled) two loop forms are
tidied:

    for (i = 0; i < (n - 1) / d + 1; ++i) S(d * i)  ->  for (i = 0; i < n; i += d) S(i)
    for (i = 0; i < u - o; ++i) S(i + o)           ->  for (i = o; i < u; ++i) S(i)

(the division may be a shift or a multiplication by a magic number), and two
versions of the same code under a pure condition (`if (C) { A; return; } A;`,
`if (C) A else A`), which the compiler makes for arrays that might overlap,
become one: once re-rolled both read the same.
"""
from .ir import Assign, Store, Var, Bin, Un, Const, Cast, Load, Select, Call, Intr, map_expr
from .simplify import simplify, repr_key


def _S():
    m = _S.__dict__.get("m")
    if m is None:              # imported late: structure imports this module
        from . import structure as m
        _S.m = m
    return m


def _L():
    from . import loops
    return loops


K = 2          # unroll factor handled (lcc -O2)


# ------------------------------------------------------------ expressions
def _pure(e):
    return not any(isinstance(x, Call) or (isinstance(x, Intr) and not x.pure) for x in e.walk())


def _vars(e):
    return {x.var for x in e.walk() if isinstance(x, Var)}


def _conv(e, ty):
    if e.ty == ty or ty not in ("i8", "i16", "i32", "i64") or e.ty not in ("i8", "i16", "i32", "i64"):
        return e
    return Cast(e, ty, signed=True)


def _subst(e, env):
    def f(x):
        if isinstance(x, Var) and x.var in env:
            return _conv(env[x.var], x.ty)
        return x
    return simplify(map_expr(e, f))


def _replace(e, key, by):
    """Replace every subexpression with repr_key == key by `by`."""
    def f(x):
        if repr_key(x) == key:
            return by
        return x
    return simplify(map_expr(e, f))


def _push_casts(e):
    """(long)(a + b) -> (long)a + (long)b for int arithmetic (no overflow of an
    index), so index expressions compare the same however they were cast."""
    def f(x):
        if isinstance(x, Cast) and x.signed and x.ty == "i64" and x.a.ty == "i32":
            a = x.a
            if isinstance(a, Bin) and a.op in ("+", "-") and a.ty == "i32":
                return Bin(a.op, f(Cast(a.a, "i64", True)), f(Cast(a.b, "i64", True)), "i64")
            if isinstance(a, Bin) and a.op in ("*", "<<") and isinstance(a.b, Const) and a.ty == "i32":
                return Bin(a.op, f(Cast(a.a, "i64", True)), Const(a.b.value, "i64"), "i64")
            if isinstance(a, Const):
                return Const(a.signed(), "i64")
        return x
    return map_expr(e, f)


AC = ("^", "|", "&", "*")       # integer only: float + and * do not reassociate


def _flat(e, op, ty, out):
    if isinstance(e, Bin) and e.op == op and e.ty == ty:
        _flat(e.a, op, ty, out)
        _flat(e.b, op, ty, out)
    else:
        out.append(e)


def _norm(x):
    from .depipe import lin_simplify
    from .ir import BitCast
    if isinstance(x, BitCast) and isinstance(x.a, Load) and x.a.ty in ("i32", "i64") and \
            x.ty == {"i32": "f32", "i64": "f64"}[x.a.ty]:
        return Load(x.a.addr, x.ty)                  # the same bits loaded as a float
    if isinstance(x, Bin) and x.ty in ("i32", "i64") and x.op == "^" and isinstance(x.b, Const) and \
            x.b.value == (1 << (64 if x.ty == "i64" else 32)) - 1:
        x = Bin("-", Const(x.b.value, x.ty), x.a, x.ty)          # ~a == -1 - a
    if isinstance(x, Bin) and x.ty in ("i32", "i64") and (x.op in ("+", "-", "<<") or
                                                         x.op == "*" and isinstance(x.b, Const)):
        return lin_simplify(x, True)
    if isinstance(x, Bin) and x.op in AC and x.ty in ("i8", "i16", "i32", "i64"):
        ops = []
        _flat(x, x.op, x.ty, ops)
        ops.sort(key=lambda o: str(repr_key(o)))
        r = ops[0]
        for o in ops[1:]:
            r = Bin(x.op, r, o, x.ty)
        return r
    return x


def canon(e):
    return map_expr(_push_casts(simplify(e)), _norm)


def arith(e):
    """Integer expression in linear normal form (~a as -1 - a): `n + (k ^ -1) + 1` -> n - k."""
    from .depipe import lin_simplify

    def f(x):
        if isinstance(x, Bin) and x.ty in ("i32", "i64") and x.op == "^" and isinstance(x.b, Const) and \
                x.b.value == (1 << (64 if x.ty == "i64" else 32)) - 1:
            return Bin("-", Const(x.b.value, x.ty), x.a, x.ty)
        return x
    return lin_simplify(map_expr(simplify(e), f))


def ckey(e):
    return str(repr_key(canon(e)))


# ------------------------------------------------------------ value ranges
def _rng(e, depth=0):
    """(lo, hi) of e read as unsigned, or None when it may wrap / is unknown."""
    from .ir import BITS
    ty = e.ty
    if ty not in ("i8", "i16", "i32", "i64") or depth > 12:
        return None
    bits = BITS[ty]
    full = (0, (1 << bits) - 1)
    if isinstance(e, Const):
        return (e.value, e.value)
    if isinstance(e, Cast) and e.a.ty in ("i8", "i16", "i32", "i64"):
        a = _rng(e.a, depth + 1) or (0, (1 << BITS[e.a.ty]) - 1)
        ba = BITS[e.a.ty]
        if bits >= ba:
            if not e.signed or a[1] < 1 << (ba - 1):
                return a
            return full
        return a if a[1] < 1 << bits else full
    if isinstance(e, Bin) and e.ty == ty:
        a = _rng(e.a, depth + 1) or full
        if e.op == "&" and isinstance(e.b, Const):
            return (0, min(a[1], e.b.value))
        if e.op == ">>u" and isinstance(e.b, Const):
            return (a[0] >> e.b.value, a[1] >> e.b.value)
        b = _rng(e.b, depth + 1) or full
        if e.op == "+":
            r = (a[0] + b[0], a[1] + b[1])
        elif e.op == "*":
            r = (a[0] * b[0], a[1] * b[1])
        elif e.op == "<<" and isinstance(e.b, Const):
            r = (a[0] << e.b.value, a[1] << e.b.value)
        else:
            return full
        return r if r[1] < 1 << bits else full
    if isinstance(e, Select):
        a, b = _rng(e.a, depth + 1), _rng(e.b, depth + 1)
        if a and b:
            return (min(a[0], b[0]), max(a[1], b[1]))
    return full


def nonneg_range(e):
    from .ir import BITS
    r = _rng(simplify(e))
    return r is not None and r[1] < 1 << (BITS[e.ty] - 1)


# ------------------------------------------------------------ k*M -> J
class Fail(Exception):
    pass


def _lin(e, ty):
    from .depipe import _lin as lin
    return lin(e, ty)


def _jsub(e, mkeys, J, k, kmkeys=frozenset()):
    """Replace c * M (c a multiple of k) by (c / k) * J, and c * X by c * J where X
    is known to equal k * M (keys in kmkeys); any other use of M fails."""
    def has(x):
        return any(str(repr_key(y)) in mkeys or str(repr_key(y)) in kmkeys for y in x.walk())
    if not has(e):
        return e
    ke = str(repr_key(e))
    if ke in kmkeys:
        return _conv(J, e.ty)
    if ke in mkeys:
        raise Fail("M outside k*M")
    if isinstance(e, (Bin, Un, Cast)) and e.ty in ("i32", "i64") and \
            (isinstance(e, Cast) or e.op in ("+", "-", "*", "<<")):
        p = _push_casts(e)
        terms, c = _lin(p, e.ty)
        bits = 64 if e.ty == "i64" else 32
        out = None
        jc = 0
        for _, (co, t) in terms.items():
            co %= 1 << bits
            sco = co - (1 << bits) if co >> (bits - 1) else co
            if sco == 0:
                continue
            kt = str(repr_key(simplify(t)))
            if kt in mkeys:
                if sco % k:
                    raise Fail("coefficient")
                jc += sco // k
                continue
            if kt in kmkeys:
                jc += sco
                continue
            if has(t):
                t = t.replace(lambda ch: _jsub(ch, mkeys, J, k, kmkeys))     # M under a load, a select...
            x = t if sco == 1 else Bin("*", t, Const(co, e.ty), e.ty)
            out = x if out is None else Bin("+", out, x, e.ty)
        if jc:
            jv = _conv(J, e.ty)
            x = jv if jc == 1 else Bin("*", jv, Const(jc % (1 << bits), e.ty), e.ty)
            out = x if out is None else Bin("+", x, out, e.ty)
        c %= 1 << bits
        if out is None:
            return Const(c, e.ty)
        if c:
            out = Bin("+", out, Const(c, e.ty), e.ty)
        return simplify(out)
    return e.replace(lambda ch: _jsub(ch, mkeys, J, k, kmkeys))


# ------------------------------------------------------------ symbolic run
def _copy_stmt(s, fn):
    if isinstance(s, Assign):
        return Assign(s.dst, fn(s.src))
    if isinstance(s, Store):
        return Store(fn(s.addr), fn(s.val), s.ty)
    raise Fail("stmt")


def _symexec(stmts):
    env, mem = {}, []

    def ev(e):
        def f(x):
            if isinstance(x, Var) and x.var in env:
                return _conv(env[x.var], x.ty)
            if isinstance(x, Load):
                ak = ckey(x.addr)
                for k, ty, v in reversed(mem):
                    if k == ak and ty == x.ty:
                        return v
            return x
        return simplify(map_expr(e, f))
    for s in stmts:
        if isinstance(s, Assign):
            env[s.dst.var] = ev(s.src)
        elif isinstance(s, Store):
            mem.append((ckey(ev(s.addr)), s.ty, ev(s.val)))
        else:
            raise Fail("stmt kind")
    final = {}
    for k, ty, v in mem:
        final[(k, ty)] = ckey(_narrow(v, ty))
    return env, final


def _narrow(e, ty):
    """e modulo 2^bits(ty): what a store of type ty keeps (ring operations only)."""
    from .ir import BITS
    if ty not in ("i8", "i16", "i32") or e.ty not in ("i8", "i16", "i32", "i64"):
        return e
    bits = BITS[ty]
    if isinstance(e, Const):
        return Const(e.value, ty)
    if isinstance(e, Cast) and e.a.ty in ("i8", "i16", "i32", "i64") and BITS[e.a.ty] >= bits and BITS[e.ty] >= bits:
        return _narrow(e.a, ty)
    if isinstance(e, Cast) and e.a.ty in ("i8", "i16", "i32", "i64") and BITS[e.a.ty] < bits:
        return Cast(e.a, ty, e.signed)
    if isinstance(e, Bin) and e.op in ("+", "-", "*", "&", "|", "^") and BITS.get(e.ty, 0) >= bits:
        return Bin(e.op, _narrow(e.a, ty), _narrow(e.b, ty), ty)
    if isinstance(e, Un) and e.op in ("-", "~") and BITS.get(e.ty, 0) >= bits:
        return Un(e.op, _narrow(e.a, ty), ty)
    if isinstance(e, Load) and BITS.get(e.ty, 0) == bits:
        return Load(e.addr, e.ty)
    if BITS.get(e.ty, 0) > bits:
        return Cast(e, ty, False)
    return e


# ------------------------------------------------------------ region matching
def _plain(n):
    L = _L()
    a = L.plain_assign(n)
    return a if a is not None and _pure(a.src) else None


def _body_stmt(n):
    """Statement of a re-rollable body: unguarded pure assignment or store."""
    S = _S()
    if not isinstance(n, S.SStmt) or n.s.guard is not None:
        return None
    s = n.s
    if isinstance(s, Assign) and isinstance(s.dst, Var) and not getattr(s, "is_call", False) and _pure(s.src):
        return s
    if isinstance(s, Store) and _pure(s.addr) and _pure(s.val) and not getattr(s, "asm", None):
        return s
    return None


def _counted_loop(n):
    """for (i = 0; i < Mp; ++i) B  ->  (ivar, Mp, [stmts]) or None"""
    S = _S()
    if not (isinstance(n, S.SLoop) and n.kind == "for" and n.init is not None and n.step is not None):
        return None
    ini, st, c = n.init, n.step, n.cond
    if not (isinstance(ini.dst, Var) and isinstance(ini.src, Const) and ini.src.value == 0):
        return None
    iv = ini.dst.var
    if not (isinstance(st.dst, Var) and st.dst.var is iv and isinstance(st.src, Bin) and st.src.op == "+"
            and isinstance(st.src.a, Var) and st.src.a.var is iv and isinstance(st.src.b, Const)
            and st.src.b.value == 1):
        return None
    if not (isinstance(c, Bin) and c.op == "<s" and isinstance(c.a, Var) and c.a.var is iv):
        return None
    stmts = []
    for x in n.body:
        s = _body_stmt(x)
        if s is None:
            return None
        if isinstance(s, Assign) and s.dst.var is iv:
            return None
        stmts.append(s)
    return iv, c.b, stmts


def _nonzero_var(c):
    """`M != 0` (or `M >u 0`) -> M"""
    c = simplify(c)
    if isinstance(c, Bin) and c.op in ("!=", ">u") and isinstance(c.b, Const) and c.b.value == 0:
        return c.a
    return None


def _tail(nodes, q, top):
    """Tail forms at nodes[q]: (tail condition, tail stmts, end, continuation)."""
    S = _S()
    n = nodes[q]
    rest = nodes[q + 1:]
    if isinstance(n, S.SStmt) and n.s.guard is not None:
        s = n.s
        if isinstance(s, Assign) and isinstance(s.dst, Var) and not getattr(s, "is_call", False):
            return n.s.guard, [Assign(s.dst, s.src)], q + 1, []
        if isinstance(s, Store):
            return n.s.guard, [Store(s.addr, s.val, s.ty)], q + 1, []
        return None
    if not isinstance(n, S.SIf):
        return None

    def stmts(ns):
        out = []
        for x in ns:
            s = _body_stmt(x)
            if s is None:
                return None
            out.append(_copy_stmt(s, lambda e: e))
        return out

    def ret_var(ns):
        if len(ns) == 1 and isinstance(ns[0], S.SReturn) and isinstance(ns[0].value, Var):
            return ns[0].value
        return None

    def ret_expr(ns):
        if len(ns) == 1 and isinstance(ns[0], S.SReturn) and ns[0].value is not None:
            return ns[0].value
        return None
    # if (TC) T
    if not n.els and n.then and not any(_L().contains_jump(x) for x in n.then):
        T = stmts(n.then)
        if T is not None:
            return n.cond, T, q + 1, []
    # if (!TC) return v; return E;   /  if (c) return v; else return E;
    others = n.els if n.els else rest
    if (n.els and not rest) or (not n.els and len(rest) == 1):
        for a, b, tc in ((n.then, others, Un("!", n.cond, "b")), (others, n.then, n.cond)):
            v, E = ret_var(a), ret_expr(b)
            if v is not None and E is not None and repr_key(E) != repr_key(v):
                return simplify(tc), [Assign(v, E)], len(nodes), [S.SReturn(v)]
    # if (!TC) return; T   (end of a void function)
    if not n.els and len(n.then) == 1 and isinstance(n.then[0], S.SReturn) and n.then[0].value is None:
        tl = list(rest)
        cont = []
        if tl and isinstance(tl[-1], S.SReturn) and tl[-1].value is None:
            tl = tl[:-1]
            cont = [S.SReturn(None)]
        elif not top:
            return None
        T = stmts(tl)
        if T:
            return simplify(Un("!", n.cond, "b")), T, len(nodes), cont
    return None


DEBUG = __import__("os").environ.get("E2KDC_REROLL_DEBUG")


def _no(why):
    if DEBUG:
        print("reroll: no:", why)
    return False


def _env_run(stmts, env):
    env = dict(env)
    for a in stmts:
        env[a.dst.var] = _subst(a.src, env)
    return env


def _decide(e, M, mk):
    """Selects whose condition is fixed by M != 0, M >= 0."""
    def f(x):
        if isinstance(x, Select):
            c = simplify(x.c)
            t = None
            if isinstance(c, Bin) and isinstance(c.b, Const) and str(repr_key(c.a)) in mk:
                v = c.b.signed()
                if c.op in ("<=s", "==") and v == 0 or c.op == "<s" and v == 1:
                    t = False
                elif c.op in (">s", "!=") and v == 0 or c.op == ">=s" and v == 1:
                    t = True
            if t is not None:
                return x.a if t else x.b
        return x
    return simplify(map_expr(e, f))


def _unmax(e):
    """M > 0 ? M : 1  /  M <= 0 ? 1 : M  ->  M"""
    e = simplify(e)
    if isinstance(e, Select):
        c = simplify(e.c)
        if isinstance(c, Bin) and isinstance(c.b, Const) and c.b.value == 0:
            if c.op == ">s" and repr_key(e.a) == repr_key(c.a) and isinstance(e.b, Const) and e.b.value == 1:
                return e.a
            if c.op == "<=s" and repr_key(e.b) == repr_key(c.a) and isinstance(e.a, Const) and e.a.value == 1:
                return e.b
    return e


def _parity(tc, M, res):
    """tc == (k*M != N) with M == N >>u 1 (k*M also written N & -k)  ->  (N as
    written in tc, the side equal to k*M), else None."""
    tc = simplify(tc)
    if not (isinstance(tc, Bin) and tc.op == "!=") or M.ty not in ("i32", "i64"):
        return None
    km = ckey(Bin("*", M, Const(K, M.ty), M.ty))
    halves = (ckey(M), ckey(res(M)))
    for x, nn in ((tc.a, tc.b), (tc.b, tc.a)):
        if nn.ty != M.ty:
            continue
        ok_half = ckey(Bin(">>u", nn, Const(1, nn.ty), nn.ty)) in halves or \
            ckey(Bin(">>u", res(nn), Const(1, nn.ty), nn.ty)) in halves
        if not ok_half:
            continue
        if ckey(x) == km:
            return nn, x
        rx = simplify(res(x))
        if isinstance(rx, Bin) and rx.op == "&" and isinstance(rx.b, Const) and \
                rx.b.signed() == -K and ckey(rx.a) in (ckey(nn), ckey(res(nn))):
            return nn, x                     # N & -2 == 2 * (N >> 1)
    return None


def _exit_values(B, ivar, Mp, loopdefs, regdefs):
    """Variables the loop body sets once, to a function of the index and of values
    the region does not change (t = 2 * i), without reading them before: their
    value after the last iteration, i = Mp - 1 (the loop ran at least once)."""
    cnt = {}
    for s in B:
        if isinstance(s, Assign):
            cnt[s.dst.var] = cnt.get(s.dst.var, 0) + 1
    last = simplify(Bin("-", Mp, Const(1, Mp.ty), Mp.ty))
    out = {}
    for s in B:
        if not isinstance(s, Assign) or cnt[s.dst.var] != 1:
            continue
        v, xs = s.dst.var, _vars(s.src)
        if ivar not in xs or v in xs or s.src.has_load() or (xs - {ivar}) & (regdefs | loopdefs):
            continue
        if _first_read(B, v):
            continue
        out[v] = _subst(s.src, {ivar: last})
    return out


def _first_read(B, v):
    """{v} when the body reads v before it assigns it, else empty."""
    for s in B:
        for e in ([s.src] if isinstance(s, Assign) else [s.addr, s.val]):
            if v in _vars(e):
                return {v}
        if isinstance(s, Assign) and s.dst.var is v:
            return set()
    return set()


def _sexprs(s):
    return [s.src] if isinstance(s, Assign) else [s.addr, s.val]


def try_reroll(nodes, p, facts, top, root, single):
    S, L = _S(), _L()
    from .ivcanon import Facts, _pretty_addrs
    X = nodes[p]
    guarded = isinstance(X, S.SIf)
    if guarded:
        # if (M != 0) { ...; for (...) B; ... } [else { ... }]
        br = []
        for side, lst in (("then", X.then), ("els", X.els)):
            idx = [j for j, x in enumerate(lst) if isinstance(x, S.SLoop)]
            if len(idx) == 1:
                br.append((side, lst, idx[0]))
        if len(br) != 1:
            return _no("no single loop in the if")
        side, LB, li = br[0]
        OB = X.els if side == "then" else X.then
        lc = X.cond if side == "then" else Un("!", X.cond, "b")
    elif isinstance(X, S.SLoop):
        # the loop alone; if it runs max(M, 1) times, M >= 1 must follow from the facts
        LB, li, OB, lc = [X], 0, [], None
    else:
        return _no("not an if / loop")
    cl = _counted_loop(LB[li])
    if cl is None:
        return _no("not a counted loop")
    ivar, Mp_raw, B = cl
    pre_L, post_L, other = [], [], []
    for lst, out in ((LB[:li], pre_L), (LB[li + 1:], post_L), (OB, other)):
        for x in lst:
            a = _plain(x)
            if a is None:
                return _no("statements around the loop")
            out.append(a)
    # ---- pre-segment, statements between the loop and the tail, the tail
    a0 = p
    while a0 > 0 and _plain(nodes[a0 - 1]) is not None:
        a0 -= 1
    pre = [_plain(x) for x in nodes[a0:p]]
    q = p + 1
    while q < len(nodes) and _plain(nodes[q]) is not None:
        q += 1
    if q >= len(nodes):
        return _no("no tail")
    between_nodes = nodes[p + 1:q]
    between = [_plain(x) for x in between_nodes]
    tl = _tail(nodes, q, top)
    if tl is None:
        return _no("tail form")
    TC, T, end, cont = tl
    if not _pure(TC):
        return _no("tail condition has side effects")
    # ---- M, and the values on both paths
    env_pre = _env_run(pre, {})
    need_m1 = False
    if guarded:
        Mr = _nonzero_var(lc)
        if Mr is None:
            return _no("guard is not M != 0")
        M = _subst(Mr, env_pre)
    else:
        Mr = Mp_raw
        Mp0 = _subst(Mp_raw, env_pre)
        M = _unmax(Mp0)
        need_m1 = repr_key(M) != repr_key(simplify(Mp0))
    mk = {str(repr_key(M)), str(repr_key(Cast(M, "i64", True))), str(repr_key(Cast(M, "i64", False)))}
    env0 = _env_run(between, _env_run(other, env_pre))
    envL = _env_run(pre_L, env_pre)
    Mp = _decide(_subst(Mp_raw, envL), M, mk)
    if ckey(Mp) != ckey(M):
        return _no("loop count is not M")
    loopdefs = L.node_defs(LB[li])
    regdefs0 = set(env0) | set(envL) | loopdefs | {a.dst.var for a in post_L + between}
    # the loop ran at least once on this path: affine values of the last iteration
    exitv = _exit_values(B, ivar, M, loopdefs, regdefs0) if (guarded or need_m1) else {}
    env1 = dict(envL)
    for v in loopdefs:
        env1.pop(v, None)
    env1.update(exitv)
    env1 = _env_run(between, _env_run(post_L, env1))
    regdefs = set(env0) | set(env1) | loopdefs
    readsB = set()
    for s in B:
        for e in _sexprs(s):
            readsB |= _vars(e)
    tdefs = {x.dst.var for x in T if isinstance(x, Assign)}
    # statements between the loop and the tail that only feed the tail (a load
    # the compiler hoisted out of it) go with the tail
    tail_uses = L.count_vars([S.SStmt(x) for x in T])["uses"]
    for v, c in L.count_vars([S.SIf(TC, [], [])])["uses"].items():
        tail_uses[v] = tail_uses.get(v, 0) + c
    cnt = L.count_vars(root)
    tail_only = set()
    for j in range(len(between) - 1, -1, -1):
        v = between[j].dst.var
        uses_v = cnt["uses"].get(v, 0)
        inner_uses = tail_uses.get(v, 0) + sum(1 for k2 in tail_only for x in [between[k2].src]
                                               for y in x.walk() if isinstance(y, Var) and y.var is v)
        if uses_v and uses_v == inner_uses and cnt["defs"].get(v, 0) == 1:
            tail_only.add(j)
    keep_between = [a for j, a in enumerate(between) if j not in tail_only]
    # assignments around the loop must not touch what the iterations read or write
    around = {a.dst.var for a in pre_L + post_L + other + between}
    if around & (loopdefs | readsB):
        return _no("assignments around the loop touch the iterations")
    stores = any(isinstance(x, Store) for x in B + T)
    for a in post_L:
        if _vars(a.src) & ((loopdefs - set(exitv)) | tdefs):
            return _no("statement after the loop reads what the iterations change")
    for a in keep_between:
        if _vars(a.src) & (loopdefs | tdefs) or (stores and a.src.has_load()):
            return _no("statement after the loop reads what the iterations change")
    # ---- helpers read by the tail: both paths agree
    used = set(_vars(TC))
    for s in T:
        for e in _sexprs(s):
            used |= _vars(e)
    helpers = ((set(env0) | set(env1)) & used) - loopdefs
    zero = Const(0, M.ty)
    mkey = repr_key(M)
    val1 = {}
    for h in helpers:
        v1 = _decide(env1.get(h, Var(h)), M, mk)
        v0 = env0.get(h, Var(h)) if guarded else v1
        # values are in terms of the variables on entry to the region
        if (_vars(v1) | _vars(v0)) & regdefs - {h}:
            return _no("helper value depends on the region")
        if ckey(_replace(v1, mkey, zero)) != ckey(_replace(v0, mkey, zero)):
            return _no("paths disagree")
        val1[h] = v1
    # ---- trip count
    single_f = Facts(single)

    def res(e):
        return single_f.resolve(e)
    TC1 = _subst(TC, val1)
    par = _parity(TC1, M, res)
    if par is None:
        return _no("tail condition is not N odd")
    Nn, kmx = par
    kmkeys = {ckey(kmx), str(repr_key(kmx)), str(repr_key(simplify(res(kmx))))}
    # N as written before the region: its variables unchanged since then
    Nraw = None
    tcr = simplify(TC)
    xdefs = (regdefs - set(env_pre)) | {a.dst.var for a in between}
    if isinstance(tcr, Bin) and tcr.op == "!=":
        for x in (tcr.a, tcr.b):
            if ckey(_subst(x, val1)) == ckey(Nn) and not (_vars(x) & xdefs):
                Nraw = x
    if Nraw is None:
        Nraw = Nn
        if _vars(Nn) & (set(env1) | loopdefs):
            return _no("count changed in the region")
    if not (single_f.nonneg(Nraw, facts) or nonneg_range(res(Nraw)) or
            single_f.nonneg(arith(res(Nraw)), facts)):
        return _no("count may be negative")
    if need_m1:
        # the loop runs max(M, 1) times: N >= 2 makes that M
        rr = single_f.refine(res(Nraw), facts)
        if not any(single_f.pos(arith(Bin("+", x, Const(-1, x.ty), x.ty)), facts) for x in (Nraw, rr)):
            return _no("M >= 1 not proved")
    Nfin = Nraw
    rN = res(Nraw)
    r = single_f.refine(rN, facts)
    simple_vars = all(single.get(v) is not None or _defs0(root, v) for v in _vars(r))
    # `n > 0 ? n : 1` under n > 0: the plain count reads better than its variable
    if any(isinstance(x, Select) for x in rN.walk()) and not any(isinstance(x, Select) for x in r.walk()) and \
            simple_vars:
        Nfin = r
    elif not isinstance(Nraw, Var):
        Nfin = arith(Nraw)
    if Nfin.ty != ivar.ty:
        Nfin = simplify(Cast(Nfin, ivar.ty, True)) if ivar.ty in ("i32", "i64") else Nfin
    # ---- one iteration from the tail
    J = Var(ivar)
    try:
        Sb = [_copy_stmt(s, lambda e: _jsub(_subst(e, val1), mk, J, K, kmkeys)) for s in T]
        # the loop body == S(k*i); ...; S(k*i + k-1)
        Bv = [_copy_stmt(s, lambda e: _subst(e, {v: x for v, x in envL.items() if v not in loopdefs}))
              for s in B]
        envB, memB = _symexec(Bv)
        seq = []
        for r_ in range(K):
            jx = Bin("+", Bin("*", Var(ivar), Const(K, ivar.ty), ivar.ty), Const(r_, ivar.ty), ivar.ty)
            seq += [_copy_stmt(s, lambda e: _subst(e, {ivar: simplify(jx)})) for s in Sb]
        envS, memS = _symexec(seq)
    except Fail as ex:
        return _no("no single iteration: %s" % ex)
    if memB != memS:
        return _no("stores differ")
    inner = L.count_vars([LB[li]])
    region_tail = L.count_vars(nodes[q:end])["uses"]
    post_uses = L.count_vars([S.SStmt(a) for a in post_L])["uses"]
    for v in set(envB) | set(envS):
        if v in envB and v in envS:
            if ckey(envB[v]) != ckey(envS[v]):
                return _no("values differ")
        elif v in envS:
            return _no("extra variable")
        else:          # temporary of the unrolled body (or its exit value feeds the tail)
            outside = cnt["uses"].get(v, 0) - inner["uses"].get(v, 0) - post_uses.get(v, 0) - \
                region_tail.get(v, 0)
            if outside or cnt["defs"].get(v, 0) != inner["defs"].get(v, 0):
                return _no("body temporary used outside")
    keep = set(envS)
    # the new body reads loop state, the index and values unchanged since the region began
    sv = set()
    for s in Sb:
        for e in _sexprs(s):
            sv |= _vars(e)
    if (sv - keep - {ivar}) & regdefs:
        return _no("new body reads region values")
    if not guarded and any(v in L.node_defs(X) for v in _vars(M)):
        return _no("count defined in the loop")
    # ---- nothing after the region reads a removed definition
    removed = (L.node_defs(X) | tdefs | {between[j].dst.var for j in tail_only}) - keep
    reg = L.count_vars(nodes[a0:end])["uses"]
    for v in removed:
        if cnt["uses"].get(v, 0) - reg.get(v, 0) > 0:
            return _no("removed value used later")
    for a in keep_between:
        if _vars(a.src) & removed:
            return _no("statement after the loop reads a removed value")
    # ---- rewrite
    loop = S.SLoop("for", Bin("<s", Var(ivar), Nfin, "b"), [S.SStmt(s) for s in Sb])
    loop.init = Assign(Var(ivar), Const(0, ivar.ty))
    loop.step = Assign(Var(ivar), Bin("+", Var(ivar), Const(1, ivar.ty), ivar.ty))
    for x in loop.body:
        _pretty_addrs(x, ivar)
    dead = set(regdefs) | {v for a in pre + between for v in _vars(a.src)} | _vars(Mr) | _vars(TC) | \
        {v for x in T for e in _sexprs(x) for v in _vars(e)}
    nodes[q:end] = cont
    for j in sorted(tail_only, reverse=True):
        del nodes[p + 1 + j]
    nodes[p] = loop
    _dce(root, dead - keep)
    return loop


def _defs0(root, v):
    return _L().count_vars(root)["defs"].get(v, 0) == 0


def _dce(root, cands):
    S, L = _S(), _L()
    changed = True
    while changed:
        changed = False
        uses = L.count_vars(root)["uses"]

        def walk(ns):
            nonlocal changed
            keep = []
            for n in ns:
                a = _plain(n)
                if a is not None and a.dst.var in cands and not uses.get(a.dst.var, 0):
                    changed = True
                    continue
                if isinstance(n, S.SIf):
                    n.then = walk(n.then)
                    n.els = walk(n.els)
                elif isinstance(n, S.SLoop):
                    n.body = walk(n.body)
                elif isinstance(n, S.SSwitch):
                    n.cases = [(k, walk(b)) for k, b in n.cases]
                keep.append(n)
            ns[:] = keep
            return ns
        walk(root)


# ------------------------------------------------------------ strided loops
def magic_udiv(e):
    """(int)((unsigned long)(unsigned)x * m >> s)  ->  (x, d) when that is x / d."""
    e = simplify(e)
    if isinstance(e, Cast) and e.ty == "i32" and e.a.ty == "i64":
        e = e.a
    if not (isinstance(e, Bin) and e.op == ">>u" and isinstance(e.b, Const) and e.ty == "i64"):
        return None
    m_ = e.a
    if not (isinstance(m_, Bin) and m_.op == "*" and isinstance(m_.b, Const)):
        return None
    x = m_.a
    if not (isinstance(x, Cast) and not x.signed and x.a.ty == "i32"):
        return None
    m, sh = m_.b.value, e.b.value
    if not (32 <= sh < 64) or m <= 1 or m >> 33:
        return None
    for d in (((1 << sh) // m), ((1 << sh) // m) + 1):
        # Granlund-Montgomery: exact for every 32-bit x when 2^s <= m*d <= 2^s + 2^(s-32)
        if d >= 2 and 0 <= m * d - (1 << sh) <= 1 << (sh - 32):
            return x.a, d
    return None


def _strided_count(N):
    """N == (n - 1) / d + 1 (unsigned division)  ->  (n, d)"""
    N = simplify(N)
    if not (isinstance(N, Bin) and N.op == "+" and isinstance(N.b, Const) and N.b.value == 1 and N.ty == "i32"):
        return None
    q_ = simplify(N.a)
    if isinstance(q_, Bin) and q_.op == ">>u" and isinstance(q_.b, Const) and q_.ty == "i32" and 1 <= q_.b.value < 31:
        x, d = q_.a, 1 << q_.b.value               # (n - 1) / 2^s
    else:
        md = magic_udiv(q_)
        if md is None:
            return None
        x, d = md
    n = simplify(Bin("+", x, Const(1, "i32"), "i32"))
    return n, d


def stride_loop(nodes, k, facts, root, single):
    """for (i = 0; i < (n - 1) / d + 1; ++i) S(d * i)  ->  for (i = 0; i < n; i += d) S(i)

    (also `i = 0; do { S; ++i; } while (i < (n - 1) / d + 1)`).  Needs n >= 1
    (the count is computed for that case) and i dead after the loop."""
    S, L = _S(), _L()
    from .ivcanon import Facts
    loop = nodes[k]
    if not isinstance(loop, S.SLoop) or loop.cond is None:
        return False
    c = loop.cond
    if not (isinstance(c, Bin) and c.op == "<s" and isinstance(c.a, Var)):
        return False
    iv = c.a.var
    f = Facts(single)
    sc = _strided_count(c.b)
    if sc is None:
        sc = _strided_count(f.resolve(c.b))
        if sc is None:
            return False
        # n read through the definition of the count: its variables must not change
        defs = L.count_vars(root)["defs"]
        if any(defs.get(v, 0) and v not in single for v in _vars(sc[0])):
            return False
    n, d = sc
    if loop.kind == "for":
        ini, st, body, first = loop.init, loop.step, loop.body, None
        if not (ini is not None and isinstance(ini.src, Const) and ini.src.value == 0 and ini.dst.var is iv):
            return False
    elif loop.kind == "dowhile" and k > 0 and loop.body:
        a = L.plain_assign(nodes[k - 1])
        if a is None or a.dst.var is not iv or not (isinstance(a.src, Const) and a.src.value == 0):
            return False
        st = L.plain_assign(loop.body[-1])
        body, first = loop.body[:-1], k - 1
        if st is None:
            return False
    else:
        return False
    if not (st is not None and st.dst.var is iv and L.is_step(st, iv) and st.src.op == "+"
            and st.src.b.value == 1):
        return False
    if any(L.contains_jump(x, True) for x in body) or any(iv in L.node_defs(x) for x in body):
        return False
    if not f.pos(n, facts):
        return False
    if any(_vars(x) & L.node_defs(loop) for x in [n]):
        return False
    # i used only inside the loop
    cnt = L.count_vars(root)["uses"]
    inner = L.count_vars([loop])["uses"]
    if cnt.get(iv, 0) != inner.get(iv, 0):
        return False
    I = Var(iv)
    mk = {str(repr_key(I)), str(repr_key(Cast(I, "i64", True)))}
    try:
        mapped = []
        for x in body:
            mapped.append(_map_node(x, lambda e: _jsub(e, mk, I, d) if _pure(e) else _fail()))
    except Fail:
        return False
    loop.kind = "for"
    loop.init = Assign(Var(iv), Const(0, iv.ty))
    loop.step = Assign(Var(iv), Bin("+", Var(iv), Const(d, iv.ty), iv.ty))
    old = _vars(loop.cond)
    loop.cond = Bin("<s", Var(iv), n, "b")
    loop.body = mapped
    if first is not None:
        del nodes[first]
    _dce(root, old - {iv})
    return True


def _fail():
    raise Fail("impure")


def offset_loop(nodes, k, facts, root, single):
    """for (i = 0; i < u - o; ++i) S(i + o)  ->  for (i = o; i < u; ++i) S(i)
    (o loop-invariant, every use of i comes with + o, i dead after the loop)."""
    S, L = _S(), _L()
    from .ivcanon import Facts
    loop = nodes[k]
    if not (isinstance(loop, S.SLoop) and loop.kind == "for" and loop.init is not None and loop.step is not None):
        return False
    c = loop.cond
    if not (isinstance(c, Bin) and c.op == "<s" and isinstance(c.a, Var) and c.a.ty == "i32"):
        return False
    iv = c.a.var
    ini, st = loop.init, loop.step
    if not (ini.dst.var is iv and isinstance(ini.src, Const) and ini.src.value == 0 and L.is_step(st, iv)
            and st.src.op == "+" and st.src.b.value == 1):
        return False
    N = c.b
    defs = L.count_vars(root)["defs"]
    if isinstance(N, Var) and N.var in single:
        N = Facts(single).resolve(N)
        if any(defs.get(v, 0) and v not in single for v in _vars(N)):
            return False
    N = arith(N)
    if N.ty != "i32":
        return False
    terms, _ = _lin(N, "i32")
    neg = [(t, co) for _, (co, t) in terms.items() if (co % (1 << 32)) == (1 << 32) - 1]
    pos_ = [(t, co) for _, (co, t) in terms.items() if (co % (1 << 32)) != (1 << 32) - 1]
    if len(neg) != 1 or not pos_:
        return False
    o = neg[0][0]
    if not isinstance(o, Var) or o.var in L.node_defs(loop) or o.var is iv:
        return False
    U = arith(Bin("+", N, o, "i32"))
    body = loop.body
    if any(L.contains_jump(x, True) for x in body):
        return False
    cnt = L.count_vars(root)["uses"]
    if cnt.get(iv, 0) != L.count_vars([loop])["uses"].get(iv, 0):
        return False
    ikeys = {str(repr_key(Var(iv))): "i32", str(repr_key(Cast(Var(iv), "i64", True))): "i64"}
    okeys = {"i32": str(repr_key(o)), "i64": str(repr_key(Cast(o, "i64", True)))}

    def osub(e):
        if not any(str(repr_key(x)) in ikeys for x in e.walk()):
            return e
        if str(repr_key(e)) in ikeys:
            raise Fail("bare index")
        if isinstance(e, (Bin, Un, Cast)) and e.ty in ("i32", "i64") and \
                (isinstance(e, Cast) or e.op in ("+", "-", "*", "<<")):
            p_ = _push_casts(e)
            tm, cc = _lin(p_, e.ty)
            bits = 64 if e.ty == "i64" else 32
            ik = [k_ for k_, (co, t) in tm.items() if str(repr_key(simplify(t))) in ikeys]
            if len(ik) != 1:
                if ik:
                    raise Fail("index twice")
                return e.replace(osub)
            alpha = tm[ik[0]][0] % (1 << bits)
            ok_ = [k_ for k_, (co, t) in tm.items() if str(repr_key(simplify(t))) == okeys[e.ty]]
            if len(ok_) != 1 or tm[ok_[0]][0] % (1 << bits) != alpha:
                raise Fail("index without the offset")
            out = None
            for k_, (co, t) in tm.items():
                if k_ == ok_[0]:
                    continue
                co %= 1 << bits
                if co == 0:
                    continue
                if k_ != ik[0] and any(str(repr_key(x)) in ikeys for x in t.walk()):
                    t = t.replace(osub)
                x = t if co == 1 else Bin("*", t, Const(co, e.ty), e.ty)
                out = x if out is None else Bin("+", out, x, e.ty)
            cc %= 1 << bits
            if cc:
                out = Bin("+", out, Const(cc, e.ty), e.ty)
            return simplify(out)
        return e.replace(osub)
    try:
        mapped = [_map_node(x, lambda e: osub(e) if _pure(e) else _fail()) for x in body]
    except Fail:
        return False
    from .ivcanon import _pretty_addrs
    for x in mapped:
        _pretty_addrs(x, iv)
    old = _vars(loop.cond)
    loop.init = Assign(Var(iv), o)
    loop.cond = Bin("<s", Var(iv), U, "b")
    loop.body = mapped
    _dce(root, old - {iv})
    return True


def _map_node(n, fn):
    """Copy of statement node n with every expression mapped (SStmt / SIf only)."""
    S = _S()
    if isinstance(n, S.SStmt):
        s = n.s
        if isinstance(s, Assign) and isinstance(s.dst, Var):
            ns = Assign(s.dst, fn(s.src), fn(s.guard) if s.guard is not None else None)
            for kk, vv in s.__dict__.items():
                if kk not in ("dst", "src", "guard"):
                    ns.__dict__[kk] = vv
            return S.SStmt(ns)
        if isinstance(s, Store):
            return S.SStmt(Store(fn(s.addr), fn(s.val), s.ty, fn(s.guard) if s.guard is not None else None))
        raise Fail("stmt")
    if isinstance(n, S.SIf):
        return S.SIf(fn(n.cond), [_map_node(x, fn) for x in n.then], [_map_node(x, fn) for x in n.els])
    raise Fail("node")


# ------------------------------------------------------------ versioned loops
class _Ph:
    def __init__(self, ty):
        self.name, self.ty = "_", ty


_PH = {}


def _ph(n, ty):
    """n-th loop index of a shape: the same object for both shapes compared"""
    if (n, ty) not in _PH:
        _PH[(n, ty)] = _Ph(ty)
    return _PH[(n, ty)]


def _ashape(ns, ren, cnt):
    """structural key of a statement list; loop indices compared up to renaming"""
    S = _S()
    out = []

    def key(e, r):
        return repr_key(map_expr(e, lambda x: Var(r[x.var], x.ty) if isinstance(x, Var) and x.var in r else x))

    def dkey(d, r):
        if isinstance(d, Var) and d.var in r:
            return repr_key(Var(r[d.var], d.ty))
        return repr_key(d) if d is not None else None
    for n in ns:
        if isinstance(n, S.SStmt):
            s = n.s
            # calls have no structural key (Call.key() is the object's id): never equal
            if any(not _pure(e) for e in s.exprs()) or getattr(s, "is_call", False) or \
                    getattr(s, "asm", None):
                return None
            out.append((type(s).__name__, dkey(s.dst if isinstance(s, Assign) else None, ren),
                        tuple(key(e, ren) for e in s.exprs()), getattr(s, "ty", None)))
        elif isinstance(n, S.SReturn):
            if n.value is not None and not _pure(n.value):
                return None
            out.append(("ret", key(n.value, ren) if n.value is not None else None))
        elif isinstance(n, S.SIf):
            if not _pure(n.cond):
                return None
            a, b = _ashape(n.then, ren, cnt), _ashape(n.els, ren, cnt)
            if a is None or b is None:
                return None
            out.append(("if", key(n.cond, ren), a, b))
        elif isinstance(n, S.SLoop):
            r2 = dict(ren)
            if n.init is not None and isinstance(n.init.dst, Var):
                r2[n.init.dst.var] = _ph(cnt[0], n.init.dst.var.ty)
                cnt[0] += 1
            sh = _ashape(n.body, r2, cnt)
            if sh is None:
                return None
            if n.cond is not None and not _pure(n.cond):
                return None
            steps = tuple((dkey(st.dst, r2), key(st.src, r2)) for st in (n.init, n.step) if st is not None)
            out.append(("loop", n.kind, key(n.cond, r2) if n.cond is not None else None, steps, tuple(sh)))
        else:
            return None
    return tuple(out)


def merge_versions(nodes, top):
    """if (C) { A; return X; } A; return X;  ->  A; return X;   (C pure: the two
    versions of a loop the compiler made, e.g. for possibly aliasing arrays,
    read the same once both are re-rolled)."""
    S = _S()
    changed = False
    k = 0
    while k < len(nodes):
        n = nodes[k]
        if isinstance(n, S.SIf) and not n.els and n.then and isinstance(n.then[-1], S.SReturn) and \
                _pure(n.cond):
            A = n.then[:-1]
            rest = nodes[k + 1:]
            rv = n.then[-1].value
            if rest and isinstance(rest[-1], S.SReturn):
                B, rb = rest[:-1], rest[-1].value
                same_ret = (rv is None and rb is None) or (rv is not None and rb is not None and
                                                           repr_key(rv) == repr_key(rb))
            else:
                B, same_ret = rest, rv is None and top
            if same_ret and A and len(A) == len(B) and _ashape(A, {}, [0]) is not None and \
                    _ashape(A, {}, [0]) == _ashape(B, {}, [0]):
                del nodes[k]
                changed = True
                continue
        if isinstance(n, S.SIf) and n.els and n.then and _pure(n.cond) and \
                _ashape(n.then, {}, [0]) is not None and _ashape(n.then, {}, [0]) == _ashape(n.els, {}, [0]):
            nodes[k:k + 1] = n.then                 # if (C) A else A
            changed = True
            continue
        if isinstance(n, S.SIf):
            changed |= merge_versions(n.then, False)
            changed |= merge_versions(n.els, False)
        elif isinstance(n, S.SLoop):
            changed |= merge_versions(n.body, False)
        k += 1
    return changed


# ------------------------------------------------------------ driver
def _walk(nodes, facts, top, root, st):
    S, L = _S(), _L()
    from .ivcanon import single_defs
    k = 0
    local = list(facts)
    while k < len(nodes):
        n = nodes[k]
        if isinstance(n, (S.SIf, S.SLoop)):
            try:
                ok = try_reroll(nodes, k, local, top, root, st["single"])
            except Fail:
                ok = False
            if ok:
                st["single"] = single_defs(root)
                st["n"] += 1
                k = nodes.index(ok)      # the clean-up may have removed statements before it
                continue
        n = nodes[k]
        if isinstance(n, S.SLoop):
            if stride_loop(nodes, k, local, root, st["single"]) or \
                    offset_loop(nodes, k, local, root, st["single"]):
                st["n"] += 1
                k = nodes.index(n)
        d = L.node_defs(n)
        if isinstance(n, S.SIf):
            _walk(n.then, local + [n.cond], False, root, st)
            _walk(n.els, local + [Un("!", n.cond, "b")], False, root, st)
        elif isinstance(n, S.SLoop):
            _walk(n.body, [f for f in local if not (_vars(f) & d)], False, root, st)
        elif isinstance(n, S.SSwitch):
            for _, b in n.cases:
                _walk(b, local, False, root, st)
        local = [f for f in local if not (_vars(f) & d)]
        if isinstance(n, S.SIf) and not n.els and n.then and S.ends_jump(n.then):
            local.append(Un("!", n.cond, "b"))
        k += 1


def reroll(body):
    from .ivcanon import single_defs
    st = {"single": single_defs(body), "n": 0}
    _walk(body, [], True, body, st)
    if merge_versions(body, True):
        _dce(body, set(_L().count_vars(body)["defs"]))
    return body
