"""Counted hardware loops -> index loops.

lcc counts pipelined loops with LSR; lift.py models that as `lc` and the loop
comes out as

    lc = N; p = P0; q = Q0;
    do { ... *p ... *(q + 4) ...; p += 8; q -= 4; --lc; } while (lc != 0);

Here every pointer-like induction variable is an affine function of the
iteration number, so the loop is rewritten with one index

    for (i = 0; i < N; ++i) { ... ((int *)P0)[2 * i] ... }       (N >= 1 proven)
    i = 0; do { ...; ++i; } while (i < N);                        (otherwise)

Exact: with lc_init = N >= 1 the counter is N - i after i iterations, and the
loop never ran with N == 0 (LSR lcnt is max(n, 1) in lcc output).  Pointers are
substituted only when they are dead after the loop.

Proving N >= 1 uses facts: conditions of enclosing ifs, negations of earlier
early exits (`if (n <= 0) return`), select conditions (`n > 0 ? n : 1`) and
single definitions of variables.
"""
from .ir import Assign, Var, Bin, Un, Const, Cast, Load, Select, Store, map_expr
from .simplify import simplify, repr_key


def _S():
    m = _S.__dict__.get("m")
    if m is None:              # imported late: structure imports this module
        from . import structure as m
        _S.m = m
    return m


def _L():
    from . import loops as L
    return L


# ------------------------------------------------------------ facts / positivity
def _lin(e, ty):
    from .depipe import _lin as lin
    return lin(e, ty)


def _strip(e):
    while isinstance(e, Cast) and e.a.ty in ("i8", "i16", "i32", "i64") and e.ty in ("i32", "i64"):
        e = e.a
    return e


class Facts:
    def __init__(self, single):
        self.single = single          # var -> src of its only (plain) definition

    def resolve(self, e, depth=0):
        """Var with a single definition -> its source (a few levels)."""
        while isinstance(e, Var) and e.var in self.single and depth < 4:
            e = self.single[e.var]
            depth += 1
        return e

    def conjuncts(self, f, depth=0):
        f = self.resolve(simplify(f))
        if isinstance(f, Bin) and f.op == "&&" and depth < 6:
            return self.conjuncts(f.a, depth + 1) + self.conjuncts(f.b, depth + 1)
        if isinstance(f, Un) and f.op == "!" and depth < 6:
            g = self.resolve(f.a)
            if isinstance(g, Bin) and g.op == "||":       # !(a || b) = !a && !b
                return self.conjuncts(Un("!", g.a, "b"), depth + 1) + self.conjuncts(Un("!", g.b, "b"), depth + 1)
            if not (isinstance(g, Var)):
                h = simplify(Un("!", g, "b"))
                if repr_key(h) != repr_key(f):
                    return self.conjuncts(h, depth + 1)
        return [f]

    def ge1(self, facts, e):
        """Does some fact imply e >= 1 (signed)?"""
        e = _strip(simplify(e))
        ty = e.ty if e.ty in ("i32", "i64") else "i64"
        et, ec = _lin(e, ty)
        for f0 in facts:
            for f in self.conjuncts(f0):
                if not (isinstance(f, Bin) and f.op in (">s", ">=s", "<s", "<=s")):
                    continue
                a, b = _strip(f.a), _strip(f.b)
                if f.op in (">s", ">=s"):
                    D = Bin("-", a, b, ty)
                else:
                    D = Bin("-", b, a, ty)
                strict = f.op in (">s", "<s")
                dt, dc = _lin(simplify(D), ty)
                if set(dt) != set(et) or any(dt[k][0] != et[k][0] for k in dt):
                    continue
                c0 = dc - ec              # fact: e + c0 > 0 (or >= 0)
                bits = 64 if ty == "i64" else 32
                c0 = (c0 + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))
                if (strict and c0 <= 0) or (not strict and c0 <= -1):
                    return True
        return False

    def implies(self, facts, c):
        """Do the facts imply condition c?"""
        c = simplify(c)
        keys = {repr_key(simplify(f)) for f0 in facts for f in self.conjuncts(f0)}
        if repr_key(c) in keys:
            return True
        r = self.resolve(c)
        if isinstance(r, Bin) and r.op in (">s", ">=s") and isinstance(r.b, Const):
            k = r.b.value
            if (r.op == ">s" and k == 0) or (r.op == ">=s" and k == 1):
                return self.ge1(facts, r.a)
        if isinstance(r, Bin) and r.op in ("<=s", "<s") and isinstance(r.b, Const):
            k = r.b.value
            # x <= 0 / x < 1 is implied false by x >= 1: not what we want here
        return False

    def refine(self, e, facts, depth=0):
        """Select(c, a, b) whose condition the facts decide -> that branch."""
        if depth > 4:
            return e
        e = simplify(e)
        s = _strip(e)
        if isinstance(s, Select):
            if self.implies(facts, s.c):
                r = self.refine(s.a, facts, depth + 1)
            elif self.implies(facts, Un("!", s.c, "b")):
                r = self.refine(s.b, facts, depth + 1)
            else:
                return e
            return r if s is e else simplify(Cast(r, e.ty, e.signed)) if isinstance(e, Cast) else r
        return e

    def nonneg(self, e, facts, depth=0):
        e = simplify(e)
        if isinstance(e, Const):
            bits = 64 if e.ty == "i64" else 32
            v = e.value - (1 << bits) if e.value >> (bits - 1) else e.value
            return v >= 0
        if isinstance(e, Cast) and not e.signed and e.a.ty in ("i8", "i16", "i32") and e.ty == "i64":
            return True
        if isinstance(e, Bin) and e.op == ">>u" and isinstance(e.b, Const) and e.b.value >= 1:
            return True
        if isinstance(e, Bin) and e.op == "&" and isinstance(e.b, Const) and not e.b.value >> 30:
            return True
        return self.pos(e, facts, depth + 1)

    def pos(self, e, facts, depth=0):
        """e >= 1 ?"""
        if depth > 6:
            return False
        e = simplify(e)
        if self.ge1(facts, e):
            return True
        s = _strip(e)
        if s is not e:
            if isinstance(e, Cast) and not e.signed and s.ty == "i32":
                return self.pos(s, facts, depth + 1)     # zext of a positive int
            if isinstance(e, Cast) and e.signed:
                return self.pos(s, facts, depth + 1)
        if isinstance(s, Const):
            bits = 64 if s.ty == "i64" else 32
            v = s.value - (1 << bits) if s.value >> (bits - 1) else s.value
            return v >= 1
        if isinstance(s, Var):
            r = self.resolve(s)
            if r is not s:
                return self.pos(r, facts, depth + 1)
        if isinstance(s, Select):
            return self.pos(s.a, facts + [s.c], depth + 1) and \
                self.pos(s.b, facts + [Un("!", s.c, "b")], depth + 1)
        if isinstance(s, Bin) and s.op == "+" and isinstance(s.b, Const):
            bits = 64 if s.ty == "i64" else 32
            k = s.b.value - (1 << bits) if s.b.value >> (bits - 1) else s.b.value
            if k >= 1 and self.nonneg(s.a, facts, depth + 1):
                return True
        return False


def single_defs(nodes):
    """var -> src for variables defined exactly once, by a plain assignment."""
    L = _L()
    cnt, src = {}, {}

    def walk(ns):
        S = _S()
        for n in ns:
            if isinstance(n, S.SStmt):
                d = getattr(n.s, "dst", None)
                if isinstance(d, Var):
                    cnt[d.var] = cnt.get(d.var, 0) + (2 if n.s.guard is not None else 1)
                    a = L.plain_assign(n)
                    if a is not None:
                        src[d.var] = a.src
            elif isinstance(n, S.SIf):
                walk(n.then)
                walk(n.els)
            elif isinstance(n, S.SLoop):
                for st in (n.init, n.step):
                    if st is not None and isinstance(st.dst, Var):
                        cnt[st.dst.var] = cnt.get(st.dst.var, 0) + 1
                walk(n.body)
            elif isinstance(n, S.SSwitch):
                for _, b in n.cases:
                    walk(b)
    walk(nodes)
    return {v: e for v, e in src.items() if cnt.get(v) == 1}


# ------------------------------------------------------------ the rewrite
def _conj(e):
    if isinstance(e, Bin) and e.op == "&&":
        return _conj(e.a) + _conj(e.b)
    return [e]


def _step_of(st):
    """(var, signed constant step) for `v = v + c`, else None."""
    L = _L()
    if st is None or not L.is_step(st):
        return None
    e = st.src
    bits = 64 if e.ty == "i64" else 32
    c = e.b.value % (1 << bits)
    c = c - (1 << bits) if c >> (bits - 1) else c
    return st.dst.var, (c if e.op == "+" else -c)


def index_form(addr, size, ivar):
    """base + k*i + c  ->  base + (k/size*i + c/size) * size  when it divides,
    so the emitter prints ((T *)base)[idx]."""
    from .depipe import lin_simplify
    if addr.ty != "i64":
        return addr
    terms, c = _lin(addr, "i64")
    if not any(ivar in {x.var for x in t.walk() if isinstance(x, Var)} for _, (co, t) in terms.items()):
        return addr
    base = None
    for k, (co, t) in terms.items():
        if co == 1 and isinstance(t, Var) and t.var is not ivar and getattr(t.var, "pointer", False):
            base = k
            break
    if base is None:
        for k, (co, t) in terms.items():
            if co == 1 and isinstance(t, Var) and t.var is not ivar:
                base = k
                break
    if base is None:
        return addr
    c = c - (1 << 64) if c >> 63 else c
    idx = Const(0, "i64") if c % size else Const(c // size, "i64")
    if c % size:
        return addr
    for k, (co, t) in terms.items():
        if k == base:
            continue
        co = co - (1 << 64) if co >> 63 else co
        if co % size:
            return addr
        q = co // size
        tt = t if q == 1 else Bin("*", t, Const(q % (1 << 64), "i64"), "i64")
        idx = Bin("+", tt, idx, "i64") if not (isinstance(idx, Const) and idx.value == 0) else tt
    idx = lin_simplify(idx)
    # all terms sign-extended ints: compute the index as int (a2 - 1 - i, not (long)(a2 - 1) - i)
    it, ic = _lin(idx, "i64")
    if all(isinstance(t, Cast) and t.signed and t.a.ty == "i32" for _, (co, t) in it.items()) and \
            -(1 << 31) <= (ic - (1 << 64) if ic >> 63 else ic) < (1 << 31):
        e32 = None
        for _, (co, t) in it.items():
            co = co - (1 << 64) if co >> 63 else co
            tt = t.a if co == 1 else Bin("*", t.a, Const(co % (1 << 32), "i32"), "i32")
            e32 = tt if e32 is None else Bin("+", e32, tt, "i32")
        icc = ic - (1 << 64) if ic >> 63 else ic
        if icc:
            e32 = Bin("+", e32, Const(icc % (1 << 32), "i32"), "i32")
        if e32 is not None:
            idx = Cast(lin_simplify(e32), "i64", signed=True)
    return Bin("+", terms[base][1], Bin("*", idx, Const(size, "i64"), "i64"), "i64")


def _load_fixer(ivar):
    from .ir import BITS

    def fix(e):
        if isinstance(e, Load) and e.ty in BITS and BITS[e.ty] >= 8:
            na = index_form(e.addr, BITS[e.ty] // 8, ivar)
            if na is not e.addr:
                return Load(na, e.ty, getattr(e, "signed", False), getattr(e, "spec", False))
        return e
    return fix


def _pretty_addrs(node, ivar):
    """Rewrite load/store addresses that use the index into array form."""
    from .ir import BITS
    fix = _load_fixer(ivar)
    S = _S()
    L = _L()
    if isinstance(node, S.SStmt):
        node.s.map(lambda e: map_expr(e, fix))
        if isinstance(node.s, Store) and node.s.ty in BITS:
            node.s.addr = index_form(node.s.addr, BITS[node.s.ty] // 8, ivar)
    elif isinstance(node, S.SIf):
        node.cond = map_expr(node.cond, fix)
        for x in node.then + node.els:
            _pretty_addrs(x, ivar)
    elif isinstance(node, S.SLoop):
        if node.cond is not None:
            node.cond = map_expr(node.cond, fix)
        for x in node.body:
            _pretty_addrs(x, ivar)
    elif isinstance(node, S.SSwitch):
        for _, b in node.cases:
            for x in b:
                _pretty_addrs(x, ivar)
    elif isinstance(node, S.SReturn) and node.value is not None:
        node.value = map_expr(node.value, fix)


def _find_init(nodes, k, v, body_defs, consts=None):
    """Index of the plain `v = X` before nodes[k] with nothing in between touching v
    or the variables of X; None otherwise."""
    L = _L()
    for j in range(k - 1, -1, -1):
        a = L.plain_assign(nodes[j])
        if a is not None and a.dst.var is v:
            # a variable of X redefined later: use its constant value at the init if known
            env = {}
            for x in L.expr_vars(a.src):
                for jj in range(j - 1, -1, -1):
                    b = L.plain_assign(nodes[jj])
                    if b is not None and b.dst.var is x:
                        if isinstance(b.src, Const):
                            env[x] = b.src
                        break
                    if x in L.node_defs(nodes[jj]) or L.contains_jump(nodes[jj]):
                        break
                else:
                    if consts and x in consts:          # constant on entry to this list
                        env[x] = consts[x]
            if env:
                a.src = simplify(L.subst_map(a.src, env))
            xs = L.expr_vars(a.src)
            if xs & body_defs:
                return None
            if any(L.node_defs(nodes[m]) & (xs | {v}) or L.uses_in(nodes[m], v) for m in range(j + 1, k)):
                return None
            return j
        if L.mentions(nodes[j], v) or L.contains_jump(nodes[j]):
            return None
    return None


def _fresh_name(ctx_names, base="i"):
    for n in [base, "j", "k"] + ["i%d" % d for d in range(2, 50)]:
        if n not in ctx_names:
            return n
    return base


def canon_loop(nodes, k, facts, F_, ctx, names, consts=None):
    S, L = _S(), _L()
    loop = nodes[k]
    if not (isinstance(loop, S.SLoop) and loop.kind == "dowhile" and loop.cond is not None):
        return False
    terms = _conj(loop.cond)
    lcv = None
    for t in terms:
        if isinstance(t, Bin) and t.op == "!=" and isinstance(t.a, Var) and isinstance(t.b, Const) \
                and t.b.value == 0 and t.a.var.reg == "lc":
            lcv = t.a.var
            lct = t
    if lcv is None:
        return False
    body = loop.body
    if any(L.contains_jump(n) for n in body):
        return False
    steps = [(q, _step_of(L.plain_assign(n))) for q, n in enumerate(body)]
    lsteps = [q for q, st in steps if st and st[0] is lcv and st[1] == -1]
    if len(lsteps) != 1:
        return False
    sq = lsteps[0]
    # lc used only by its step and the condition (nowhere else in the function)
    if ctx.uses.get(lcv, 0) != 2 or ctx.defs.get(lcv, 0) != 2:
        return False
    body_defs = set()
    for n in body:
        body_defs |= L.node_defs(n)
    j = _find_init(nodes, k, lcv, body_defs, consts)
    if j is None:
        return False
    N = nodes[j].s.src
    # index type: 32-bit when the count comes from a 32-bit value
    Ns = _strip(N)
    ity = "i32" if Ns.ty == "i32" else "i64"
    Nc = Ns if ity == "i32" else N
    Nc = Facts(single_defs(ctx.root)).refine(Nc, facts)
    # induction pointers: one top-level step in the body, initialised right before
    ivs = {}
    for q, st in steps:
        if not st or st[0] is lcv:
            continue
        p, c = st
        if [qq for qq, s2 in steps if s2 and s2[0] is p] != [q]:
            continue
        if sum(1 for n in body if p in L.node_defs(n)) != 1:
            continue
        if ctx.defs.get(p, 0) != 2:
            continue
        jp = _find_init(nodes, k, p, body_defs - {p}, consts)
        if jp is None:
            continue
        inside = L.uses_in(loop, p)
        if ctx.uses.get(p, 0) != inside:      # used after the loop: keep it
            continue
        if p.ty not in ("i32", "i64"):
            continue
        ivs[p] = (q, c, jp, nodes[jp].s.src)
    # ---- rewrite
    old_ty, old_name = lcv.ty, lcv.name
    lcv.ty = ity
    lcv.name = _fresh_name(names)
    lcv.fixed_name = lcv.name          # keep it through later renaming
    names.add(lcv.name)
    I = Var(lcv, ity)

    def iexpr(off, ty):
        e = I if off == 0 else Bin("+", I, Const(off, ity), ity)
        if ty != ity:
            e = Cast(e, ty, signed=True)
        return e

    def sub_iv(n, after):
        for p, (q, c, jp, P0) in ivs.items():
            off = 1 if (after is None or q < after) else 0
            e = Bin("+", P0, Bin("*", iexpr(off, p.ty), Const(c % (1 << 64 if p.ty == "i64" else 1 << 32), p.ty), p.ty), p.ty)
            L.subst_node(n, p, e)

    new_body = []
    for q, n in enumerate(body):
        if q == sq or any(q == ivq for ivq, _, _, _ in ivs.values()):
            continue
        sub_iv(n, q)
        new_body.append(n)
    # condition: evaluated after every step of the iteration (i counted already)
    cond_terms = []
    for t in terms:
        if t is lct:
            cond_terms.append(Bin("<s", I, Nc, "b"))
            continue
        tt = t
        for p, (q, c, jp, P0) in ivs.items():
            e = Bin("+", P0, Bin("*", iexpr(0, p.ty), Const(c % (1 << 64 if p.ty == "i64" else 1 << 32), p.ty), p.ty), p.ty)
            tt = L.subst_map(tt, {p: e})
        cond_terms.append(simplify(tt))
    cond = cond_terms[0]
    for t in cond_terms[1:]:
        cond = Bin("&&", cond, t, "b")
    step = Assign(I, Bin("+", I, Const(1, ity), ity))
    init = Assign(I, Const(0, ity))
    for x in new_body:
        _pretty_addrs(x, lcv)
    cond = map_expr(cond, _load_fixer(lcv))
    # remove inits (highest index first)
    for jj in sorted([j] + [jp for (_, _, jp, _) in ivs.values()], reverse=True):
        del nodes[jj]
    k = nodes.index(loop)
    only_counter = len(cond_terms) == 1
    if only_counter and Facts(single_defs(ctx.root)).pos(Nc, facts):
        loop.kind = "for"
        loop.init, loop.step, loop.cond, loop.body = init, step, cond, new_body
    else:
        loop.body = new_body + [S.SStmt(step)]
        loop.cond = cond
        nodes.insert(k, S.SStmt(init))
    ctx.refresh()
    return True


# ------------------------------------------------------------ driver (top-down with facts)
def _walk(nodes, facts, ctx, names, single, consts=None):
    """Top-down: facts (conditions known true) and constants known at the start
    of `nodes` are carried into nested statement lists."""
    S, L = _S(), _L()
    k = 0
    local = list(facts)
    cst = dict(consts or {})
    at_entry = dict(cst)            # _find_init checks nothing in the list redefines these

    def kill(d):
        for v in list(cst):
            if v in d:
                del cst[v]
    while k < len(nodes):
        n = nodes[k]
        d = L.node_defs(n)
        if isinstance(n, S.SLoop):
            if canon_loop(nodes, k, local, None, ctx, names, at_entry):
                single.clear()
                single.update(single_defs(ctx.root))
                k = nodes.index(n)
                n = nodes[k]
                d = L.node_defs(n)
            inner = [f for f in local if not (L.expr_vars(f) & L.node_defs(n))]
            inner_c = {v: c for v, c in cst.items() if v not in L.node_defs(n)}
            _walk(n.body, inner, ctx, names, single, inner_c)
        elif isinstance(n, S.SIf):
            _walk(n.then, local + [n.cond], ctx, names, single, dict(cst))
            _walk(n.els, local + [Un("!", n.cond, "b")], ctx, names, single, dict(cst))
        elif isinstance(n, S.SSwitch):
            for _, b in n.cases:
                _walk(b, local, ctx, names, single, dict(cst))
        local = [f for f in local if not (L.expr_vars(f) & d)]
        kill(d)
        a = L.plain_assign(n)
        if a is not None and isinstance(a.src, Const):
            cst[a.dst.var] = a.src
        if isinstance(n, S.SIf) and not n.els and n.then and S.ends_jump(n.then):
            local.append(Un("!", n.cond, "b"))
        k += 1


def canon_counted(nodes, ctx, var_names):
    names = set(var_names)
    _walk(nodes, [], ctx, names, single_defs(nodes))
    return nodes
