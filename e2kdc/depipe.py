"""Undo software pipelining of single-block loops.

lcc pipelines loops with rotating registers: a value is computed in one
iteration and consumed `d` iterations later through a chain of copies
(created in the IR by the %b[] / %pred rotation model of lift.py), and the
prologue precomputes the values for the first iterations.  In the IR that
looks like

    v13 = *(p + 8); v11 = *(p + 16); v14 = *(p + 24); q = p + 32; r = p + 40;
    do { s += v10; v11' = v14; v14' = *q; q' = r; r += 8; v13' = v11; v10' = v13; ... }

Here every chained variable is a function of the induction variable r:
v10 == *(r - 40) at the start of every iteration.  We

  1. find induction variables  iv = iv + c  (one unguarded def in the block);
  2. derive a closed form CF(x) (expression of ivs and loop invariants, valid at
     the start of every iteration) for each variable x whose end-of-iteration value
     is such an expression shifted by one iteration;
  3. accept CF(x) only if it reproduces the value the preheader put into x
     (this is what proves the prologue fills the pipeline that way);
  4. rewrite every read of such x (in the loop and in its exit block) by its
     closed form, with ivs adjusted for updates already done at that point.

The copy chains and the prologue loads become dead and are removed by DCE.
Finally an iv whose every use carries the same offset is rebased
(r - 40 -> r, preheader r = p + 40 -> r = p).

Loads may enter closed forms only if neither the loop nor the preheader writes
memory or calls anything.
"""
from .ir import Assign, Var, Bin, Const, Load, Store, Branch, map_expr
from .simplify import simplify, repr_key
from .analysis import has_call, collect


def _subst(e, fn):
    return map_expr(e, lambda x: fn(x) if isinstance(x, Var) else x)


def _vars(e):
    return {x.var for x in e.walk() if isinstance(x, Var)}


def _key(e):
    return repr_key(simplify(e))


# ------------------------------------------------------------ linear normal form
def _lin(e, ty):
    """e as {term_key: (coeff, term)} plus constant, over + - * const, << const, neg."""
    from .ir import Un, Cast
    bits = 64 if ty == "i64" else 32
    if isinstance(e, Const):
        return {}, e.value
    if isinstance(e, Bin) and e.ty == ty and e.op in ("+", "-"):
        a, ca = _lin(e.a, ty)
        b, cb = _lin(e.b, ty)
        sg = 1 if e.op == "+" else -1
        out = dict(a)
        for k, (c, t) in b.items():
            c0 = out.get(k, (0, t))[0]
            out[k] = (c0 + sg * c, t)
        return out, ca + sg * cb
    if isinstance(e, Bin) and e.ty == ty and e.op in ("*", "<<") and isinstance(e.b, Const):
        m = e.b.value if e.op == "*" else (1 << e.b.value)
        a, ca = _lin(e.a, ty)
        return {k: (c * m, t) for k, (c, t) in a.items()}, ca * m
    if isinstance(e, Un) and e.op == "-" and e.ty == ty:
        a, ca = _lin(e.a, ty)
        return {k: (-c, t) for k, (c, t) in a.items()}, -ca
    return {repr_key(e): (1, e)}, 0


def lin_simplify(e, canonical=False):
    """t1*c1 + t2*c2 + ... + c  for integer +/-/* trees (terms in order of first
    appearance, or sorted when canonical); else simplify(e)."""
    e = simplify(e)
    if not isinstance(e, Bin) or e.op not in ("+", "-", "*", "<<") or e.ty not in ("i32", "i64"):
        return e
    ty = e.ty
    bits = 64 if ty == "i64" else 32
    terms, c = _lin(e, ty)
    out = None
    order = sorted(terms, key=lambda k: str(k)) if canonical else \
        sorted(terms, key=lambda k: (terms[k][0] % (1 << bits)) >> (bits - 1))   # positives first
    for k in order:
        co, t = terms[k]
        co %= 1 << bits
        if co == 0:
            continue
        sco = co - (1 << bits) if co >> (bits - 1) else co
        x = t if sco == 1 else Bin("*", t, Const(co, ty), ty)
        if sco == -1:
            out = Bin("-", out, t, ty) if out is not None else Bin("*", t, Const(co, ty), ty)
            continue
        out = x if out is None else Bin("+", out, x, ty)
    c %= 1 << bits
    if out is None:
        return Const(c, ty)
    if c:
        out = Bin("+", out, Const(c, ty), ty)
    return simplify(out)


def _lkey(e):
    """Comparison key modulo linear arithmetic (applied at every level)."""
    return repr_key(map_expr(simplify(e), lambda x: lin_simplify(x, True) if isinstance(x, Bin) else x))



def _calls(stmts):
    return any(has_call(e) for s in stmts for e in s.exprs())


def _writes_mem(stmts):
    for s in stmts:
        if isinstance(s, Store):
            return True
        if any(has_call(e) for e in s.exprs()):
            return True
    return False


def _shift(e, ivs, delta_sign, which=None):
    """iv -> iv + sign*step for ivs (optionally only those in `which`)."""
    def f(x):
        v = x.var
        if v in ivs and (which is None or v in which):
            st = ivs[v]
            op = "+" if delta_sign > 0 else "-"
            return Bin(op, Var(v, x.ty), st, x.ty)
        return x
    return map_expr(simplify(_subst(e, f)), lambda x: lin_simplify(x) if isinstance(x, Bin) else x)


def collapse_loop(F, B):
    preds = [p for p in B.preds if p != B.addr]
    if len(preds) != 1 or B.addr not in B.succs:
        return False
    P = F.blocks[preds[0]]
    stmts = B.stmts
    # ---- symbolic end-of-iteration values (in start-of-iteration terms)
    env = {}
    opaque = set()
    defpos = {}            # var -> list of stmt indices defining it in B
    for k, s in enumerate(stmts):
        if isinstance(s, Assign) and isinstance(s.dst, Var):
            v = s.dst.var
            defpos.setdefault(v, []).append(k)
            if s.guard is not None or has_call(s.src) or getattr(s, "is_call", False) \
                    or (_vars(s.src) & opaque):
                opaque.add(v)
                env[v] = None
                continue
            opaque.discard(v)
            env[v] = simplify(_subst(s.src, lambda x: env[x.var] if env.get(x.var) is not None else x))
    defined = set(defpos)
    # loads in closed forms: the pipelined code already reads them ahead of the
    # loop's stores, so the compiler guaranteed they do not alias; only calls block
    mem_ok = not _calls(stmts)
    # ---- induction variables: iv = iv + step, step built from loop invariants
    ivs = {}
    for v, e in env.items():
        if e is None or len(defpos[v]) != 1 or v.is_input:
            continue
        if isinstance(e, Bin) and e.op == "+" and isinstance(e.b, Var) and e.b.var is v \
                and not (isinstance(e.a, Var) and e.a.var is v):
            e = Bin("+", e.b, e.a, e.ty)            # step + iv
        if isinstance(e, Bin) and e.op in ("+", "-") and isinstance(e.a, Var) and e.a.var is v:
            st = e.b if e.op == "+" else simplify(Bin("-", Const(0, e.ty), e.b, e.ty))
            if isinstance(st, Const):
                bits = 64 if e.ty == "i64" else 32
                c = st.value - (1 << bits) if st.value >> (bits - 1) else st.value
                if c:
                    ivs[v] = Const(c & ((1 << bits) - 1), e.ty)
            elif not (_vars(st) & defined) and not st.has_load() and not has_call(st):
                ivs[v] = st
    if not ivs:
        return False
    # ---- preheader values: straight-line chain of single-predecessor blocks ending at P
    chain = [P]
    while len(chain[0].preds) == 1 and chain[0].preds[0] not in (B.addr,) and len(chain) < 8:
        q = F.blocks[chain[0].preds[0]]
        if q in chain or len(q.succs) != 1 and q.succs.count(chain[0].addr) != 1:
            break
        chain.insert(0, q)
    penv = {}
    for blk in chain:
        for s in blk.stmts:
            if isinstance(s, Assign) and isinstance(s.dst, Var):
                v = s.dst.var
                if s.guard is not None or has_call(s.src):
                    penv[v] = None
                    continue
                penv[v] = simplify(_subst(s.src, lambda x: penv[x.var] if penv.get(x.var) is not None else x))
    mem_ok = mem_ok and not any(_calls(blk.stmts) for blk in chain)
    def init_of(x):
        v = x.var
        if v in penv:
            return penv[v]
        return x

    def at_init(e):
        """Value of a closed form at iteration 0: every variable -> preheader value."""
        bad = []

        def f(x):
            r = init_of(x)
            if r is None:
                bad.append(x)
                return x
            return r
        r = simplify(_subst(e, f))
        return None if bad else r
    # ---- closed forms
    CF = {}
    changed = True
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
        for v in defined:
            if v in CF or v in ivs or v in opaque or env.get(v) is None or v.is_param:
                continue
            if v not in penv or penv[v] is None:
                continue                     # no preheader value to check against
            e = env[v]
            ok = True

            def g(x):
                nonlocal ok
                w = x.var
                if w in ivs or w not in defined:
                    return x
                if w in CF:
                    return CF[w]
                ok = False
                return x
            e2 = _subst(e, g)
            if not ok:
                continue
            if e2.has_load() and not mem_ok:
                continue
            if not (_vars(e2) & set(ivs)):
                continue                     # not driven by an iv: leave it
            cf = _shift(e2, ivs, -1)         # value at start of next iteration, back one step
            i0 = at_init(cf)
            if i0 is None or _lkey(i0) != _lkey(penv[v]):
                continue
            CF[v] = cf
            changed = True
    if not CF:
        return False
    # ---- rewrite reads
    ivpos = {v: defpos[v][0] for v in ivs}

    def to_cur(e, k):
        upd = {v for v, p in ivpos.items() if p < k}
        return _shift(e, ivs, -1, upd)

    lastdef = {v: max(ps) for v, ps in defpos.items()}
    from .ir import Store
    effects = [k for k, st in enumerate(stmts) if isinstance(st, Store) or any(has_call(e) for e in st.exprs())]
    first_eff = effects[0] if effects else len(stmts) + 1
    # A closed form with loads re-reads memory where it is substituted.  Before the
    # first store of the iteration that is the source-order position; later uses get
    # the value materialised once at the top of the iteration, and uses of the
    # next-iteration value keep their copy chain.
    materialize, keep_chain = set(), set()

    def kind(v, k):
        prior = [p for p in defpos[v] if p < k]
        if not prior:
            return "start"
        if max(prior) == lastdef[v]:
            return "end"
        return "mid"

    def classify(e, k):
        for x in e.walk():
            if isinstance(x, Var) and x.var in CF and CF[x.var].has_load() and effects:
                kd = kind(x.var, k)
                if kd == "start" and k > first_eff:
                    materialize.add(x.var)
                elif kd == "end":
                    keep_chain.add(x.var)
    for k, st in enumerate(stmts):
        if isinstance(st, Assign) and isinstance(st.dst, Var) and st.dst.var in CF and k == lastdef[st.dst.var]:
            continue
        for e in st.exprs():
            classify(e, k)
    if B.term is not None:
        for e in B.term.exprs():
            classify(e, len(stmts))

    def rewrite(e, k):
        def f(x):
            v = x.var
            if v not in CF:
                return x
            kd = kind(v, k)
            if kd == "mid" or v in keep_chain and kd == "end":
                return x                           # intermediate value / kept chain
            if v in materialize and kd == "start":
                return x                           # value set at the top of the iteration
            form = CF[v] if kd == "start" else _shift(CF[v], ivs, +1)
            return to_cur(form, k)
        return simplify(_subst(e, f))

    for k, st in enumerate(stmts):
        if isinstance(st, Assign) and isinstance(st.dst, Var) and st.dst.var in CF and k == lastdef[st.dst.var] \
                and st.dst.var not in keep_chain:
            continue                               # the chain copy: dead after the rewrite
        st.map(lambda e, k=k: rewrite(e, k))
    if B.term is not None:
        B.term.map(lambda e: rewrite(e, len(stmts)))
    for v in sorted(materialize, key=lambda v: v.name):
        stmts.insert(0, Assign(Var(v, v.ty), simplify(CF[v])))
    # exit block: values after the loop = next-iteration closed forms with current ivs
    for xa in B.succs:
        if xa == B.addr:
            continue
        X = F.blocks[xa]
        if len(X.preds) != 1:
            continue
        stale = set()

        def fx(x):
            v = x.var
            if v in CF and v not in stale and not (_vars(CF[v]) & stale):
                return CF[v]
            return x
        for s in X.stmts:
            s.map(lambda e: simplify(_subst(e, fx)))
            if isinstance(s, Assign) and isinstance(s.dst, Var):
                stale.add(s.dst.var)
        if X.term is not None:
            X.term.map(lambda e: simplify(_subst(e, fx)))
    F._depipe = getattr(F, "_depipe", []) + [(B.addr, P.addr, dict(ivs))]
    return True


def _offset(e, v):
    """e == v + c  ->  c ;  e == v -> 0 ; else None"""
    if isinstance(e, Var) and e.var is v:
        return 0
    if isinstance(e, Bin) and e.op == "+" and isinstance(e.a, Var) and e.a.var is v and isinstance(e.b, Const):
        bits = 64 if e.ty == "i64" else 32
        c = e.b.value
        return c - (1 << bits) if c >> (bits - 1) else c
    return None


def _additive_offsets(e, v, inner, in_load=False):
    """For each maximal +/- tree of e that contains v with coefficient 1:
    (offset expr made of loop-invariant terms, inside-load flag)."""
    out = []
    if isinstance(e, Var) and e.var is v:
        return [(Const(0, e.ty), in_load)]
    if isinstance(e, Bin) and e.op in ("+", "-") and e.ty in ("i32", "i64"):
        terms, c = _lin(e, e.ty)
        kv = repr_key(Var(v, e.ty))
        hit = [k for k, (co, t) in terms.items() if isinstance(t, Var) and t.var is v]
        if hit and terms[hit[0]][0] == 1:
            off = Const(c, e.ty)
            for k, (co, t) in terms.items():
                if k == hit[0] or (_vars(t) & inner):
                    continue
                off = Bin("+", off, Bin("*", t, Const(co, e.ty), e.ty), e.ty)
            out.append((lin_simplify(off), in_load))
            # v may also hide inside non-linear terms (loads...)
            for k, (co, t) in terms.items():
                if k != hit[0]:
                    out.extend(_additive_offsets(t, v, inner, in_load))
            return out
    for ch in e.children():
        out.extend(_additive_offsets(ch, v, inner, in_load or isinstance(e, Load)))
    return out


def rebase_ivs(F, B, P, ivs):
    """Shift an iv so that its most used offset disappears:
    v = p + 40; ... *(v - 40) ... (v - 32 == e)  ->  v = p; ... *v ... (v + 8 == e).
    Offsets may be symbolic (loop invariants): v12 - 2*a1  ->  v12."""
    defs, uses = collect(F)
    inner = {stmt.dst.var for stmt in B.stmts if isinstance(stmt, Assign) and isinstance(stmt.dst, Var)}
    for v in ivs:
        ds = defs.get(v, [])
        if len(ds) != 2:
            continue
        pdef = [(b, i) for b, i in ds if b is P]
        if len(pdef) != 1 or P.stmts[pdef[0][1]].guard is not None:
            continue
        pb, pi = pdef[0]
        own = defpos_of(B, v)
        weight, rep = {}, {}
        ok = True
        sites = []
        for b, i in uses.get(v, []):
            if b is B and i == own:
                continue
            if b is P and i < pi:
                ok = False
            st = b.stmts[i] if i < len(b.stmts) else b.term
            for e in st.exprs():
                for off, in_load in _additive_offsets(e, v, inner):
                    k = _lkey(off)
                    weight[k] = weight.get(k, 0) + (2 if in_load else 1)
                    rep[k] = off
            if all(st is not x for x in sites):
                sites.append(st)
        if not ok or not weight:
            continue
        best = max(weight, key=lambda k: (weight[k], -len(str(k))))
        off = rep[best]
        if isinstance(off, Const) and off.value == 0:
            continue
        ty = P.stmts[pi].dst.ty
        # the new start reads `off`: it must come after the definitions of its
        # variables in the preheader (v11 = 0 ... v15 = a1 + 4  ->  v11 = v15 after it)
        deps = {x.var for x in off.walk() if isinstance(x, Var)}
        last = max((j for j in range(pi + 1, len(P.stmts))
                    if getattr(getattr(P.stmts[j], "dst", None), "var", None) in deps), default=None)
        if last is not None:
            if any(isinstance(x, Var) and x.var is v for j in range(pi + 1, last + 1)
                   for e in P.stmts[j].exprs() for x in e.walk()):
                continue
            stmt = P.stmts.pop(pi)
            P.stmts.insert(last, stmt)
            pi = last
        P.stmts[pi].src = lin_simplify(Bin("+", P.stmts[pi].src, off, ty))
        for st in sites:
            def f(x, off=off):
                if isinstance(x, Var) and x.var is v:
                    return Bin("-", x, off, x.ty)
                return x
            st.map(lambda e: map_expr(map_expr(e, f), lambda x: lin_simplify(x) if isinstance(x, Bin) else x))


def rebase_all(F):
    for baddr, paddr, ivs in getattr(F, "_depipe", []):
        if baddr in F.blocks and paddr in F.blocks:
            rebase_ivs(F, F.blocks[baddr], F.blocks[paddr], ivs)
    F._depipe = []


def defpos_of(B, v):
    for i, s in enumerate(B.stmts):
        if isinstance(s, Assign) and isinstance(s.dst, Var) and s.dst.var is v:
            return i
    return -1


def _iv_occurrences(e, v, in_load=False):
    """(offset, inside-load-address) for each occurrence of v in e (v + c or bare v)."""
    out = []
    if isinstance(e, Bin) and e.op == "+" and isinstance(e.a, Var) and e.a.var is v and isinstance(e.b, Const):
        return [(_offset(e, v), in_load)]
    if isinstance(e, Var) and e.var is v:
        return [(0, in_load)]
    for c in e.children():
        out.extend(_iv_occurrences(c, v, in_load or isinstance(e, Load)))
    return out


def collapse_pipelines(F):
    done = False
    for B in F.block_list():
        try:
            if collapse_loop(F, B):
                done = True
        except RecursionError:
            pass
    return done
