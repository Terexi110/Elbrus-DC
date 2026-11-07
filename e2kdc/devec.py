"""Scalar form of vectorised reductions and float maps (lcc -O3).

lcc vectorises `s += f(p[i])` into 64-bit registers holding two int lanes:

    a = 0; b = 0;
    for (...) { a = __paddw(a, X); b = __paddw(b, Y); }
    r = __paddw(__paddw(a, b), ...);  h = __paddw(r, r >> 32);  ... (int)h ...

Only the sum of all lanes is ever read, so every accumulator can be replaced by
an int holding the sum of its two lanes (sums are mod 2^32 either way):

    a = 0; ...; a = a + lane0(X) + lane1(X); ...; h = a + b ...

An accumulator qualifies when all its definitions are 0 or `__paddw` of
accumulators and other values, and all its uses are operands of such `__paddw`
or of the horizontal step `h = __paddw(v, v >> 32)` whose result is only read
through `(int)h`.
"""
from .ir import Assign, Store, Var, Bin, Un, Const, Cast, BitCast, Load, Intr, Select, map_expr
from .simplify import simplify, repr_key

M32 = 0xffffffff
PF = {"pfadds": "+", "pfsubs": "-", "pfmuls": "*"}           # packed float, lane-wise
VOPS = set(PF) | {"paddw", "psubw", "pcmpgtw", "pcmpeqw"}     # lane-wise packed ops


def _S():
    m = _S.__dict__.get("m")
    if m is None:              # imported late: structure imports this module
        from . import structure as m
        _S.m = m
    return m


def _roots(nodes):
    """(holder, getter, setter) for every expression root of the AST."""
    S = _S()
    out = []

    def add_stmt(st):
        out.append(st)

    def walk(ns):
        for n in ns:
            if isinstance(n, S.SStmt):
                add_stmt(n.s)
            elif isinstance(n, S.SIf):
                out.append(n)
                walk(n.then)
                walk(n.els)
            elif isinstance(n, S.SLoop):
                out.append(n)
                for st in (n.init, n.step):
                    if st is not None:
                        add_stmt(st)
                walk(n.body)
            elif isinstance(n, S.SSwitch):
                out.append(n)
                for _, b in n.cases:
                    walk(b)
            elif isinstance(n, S.SReturn):
                out.append(n)
            elif isinstance(n, S.SRaw) and isinstance(n.text, tuple):
                out.append(n)
    walk(nodes)
    return out


def _exprs_of(h):
    S = _S()
    if isinstance(h, (S.SIf, S.SLoop)):
        return [h.cond] if h.cond is not None else []
    if isinstance(h, S.SSwitch):
        return [h.expr]
    if isinstance(h, S.SReturn):
        return [h.value] if h.value is not None else []
    if isinstance(h, S.SRaw):
        return [h.text[1]]
    return h.exprs()


def _map_root(h, fn):
    S = _S()
    if isinstance(h, (S.SIf, S.SLoop)):
        if h.cond is not None:
            h.cond = fn(h.cond)
    elif isinstance(h, S.SSwitch):
        h.expr = fn(h.expr)
    elif isinstance(h, S.SReturn):
        if h.value is not None:
            h.value = fn(h.value)
    elif isinstance(h, S.SRaw):
        h.text = (h.text[0], fn(h.text[1]))
    else:
        h.map(fn)


def _is_paddw(e):
    return isinstance(e, Intr) and e.name == "paddw" and len(e.args) == 2


def _hstep(e):
    """__paddw(v, v >> 32) -> v, else None."""
    if _is_paddw(e):
        a, b = e.args
        if isinstance(b, Bin) and b.op == ">>u" and isinstance(b.b, Const) and b.b.value == 32 \
                and repr_key(b.a) == repr_key(a):
            return a
    return None


# ------------------------------------------------------------ lanes
def lane(e, k, env=None):
    """32-bit value of lane k (0 = low) of the 64-bit packed expression e.
    env: variables already split into lanes (var -> [VarInfo lo, VarInfo hi])."""
    e = simplify(e)
    if env and isinstance(e, Var) and e.var in env and e.ty == "i64":
        lv = env[e.var][k]
        return Var(lv, "i32") if lv.ty == "i32" else BitCast(Var(lv, lv.ty), "i32")
    if isinstance(e, Intr) and e.name in PF and len(e.args) == 2:
        return BitCast(Bin(PF[e.name], lane_f(e.args[0], k, env), lane_f(e.args[1], k, env), "f32"), "i32")
    if isinstance(e, Const):
        return Const((e.value >> (32 * k)) & M32, "i32")
    if isinstance(e, Load) and e.ty == "i64":
        a = e.addr if k == 0 else Bin("+", e.addr, Const(4, "i64"), "i64")
        return Load(simplify(a), "i32")
    if isinstance(e, Bin) and e.op == "|" and e.ty == "i64":
        hi = lo = None
        for x, y in ((e.a, e.b), (e.b, e.a)):
            if isinstance(x, Bin) and ((x.op == "<<" and isinstance(x.b, Const) and x.b.value == 32) or
                                       (x.op == "*" and isinstance(x.b, Const) and x.b.value == 1 << 32)) \
                    and isinstance(y, Cast) and not y.signed and y.a.ty == "i32":
                hi, lo = x.a, y.a
        if hi is not None:
            if k == 0:
                return lo
            h = hi.a if isinstance(hi, Cast) and hi.a.ty == "i32" else Cast(hi, "i32", signed=False)
            return h
    if isinstance(e, Bin) and e.op in ("&", "|", "^") and e.ty == "i64":
        return Bin(e.op, lane(e.a, k, env), lane(e.b, k, env), "i32")
    if isinstance(e, Un) and e.op == "~" and e.ty == "i64":
        return Un("~", lane(e.a, k, env), "i32")
    if isinstance(e, Intr) and e.name in ("paddw", "psubw") and len(e.args) == 2:
        return Bin("+" if e.name == "paddw" else "-", lane(e.args[0], k, env), lane(e.args[1], k, env), "i32")
    if isinstance(e, Intr) and e.name in ("pcmpgtw", "pcmpeqw") and len(e.args) == 2:
        op = ">s" if e.name == "pcmpgtw" else "=="
        return Select(Bin(op, lane(e.args[0], k, env), lane(e.args[1], k, env), "b"),
                      Const(M32, "i32"), Const(0, "i32"), "i32")
    x = e if k == 0 else Bin(">>u", e, Const(32, "i64"), "i64")
    return Cast(x, "i32", signed=False)


def lane_f(e, k, env=None):
    """lane k of e as a float"""
    x = simplify(lane(e, k, env))
    if isinstance(x, BitCast) and x.a.ty == "f32":
        return x.a
    return BitCast(x, "f32")


def lanesum(e):
    return simplify(Bin("+", lane(e, 0), lane(e, 1), "i32"))


# ------------------------------------------------------------ the pass
def devectorize(nodes):
    S = _S()
    roots = _roots(nodes)
    # definitions and uses with their context
    defs = {}          # var -> [Assign]
    for h in roots:
        if isinstance(h, Assign) and isinstance(h.dst, Var):
            defs.setdefault(h.dst.var, []).append(h)
    cand = {v for v, ds in defs.items() if v.ty == "i64" and any(_is_paddw(d.src) for d in ds)}
    hsteps = {}        # var h defined as __paddw(v, v >> 32)
    for v, ds in defs.items():
        if len(ds) == 1 and _hstep(ds[0].src) is not None and ds[0].guard is None:
            hsteps[v] = ds[0]

    def ok_def(d, C):
        s = d.src
        if isinstance(s, Const) and s.value == 0:
            return True
        if _is_paddw(s):
            return True
        if isinstance(s, Var) and s.var in C:
            return True
        return False

    # uses: an occurrence of an accumulator must be the whole source of an accumulator
    # def (copy), a direct operand of its __paddw, or the operand of an h-step; an h
    # variable may only be read as (int)h
    def bad_uses(C, H):
        bad = set()
        for h in roots:
            acc_def = isinstance(h, Assign) and isinstance(h.dst, Var) and h.dst.var in C
            h_def = isinstance(h, Assign) and isinstance(h.dst, Var) and h.dst.var in H
            for e in _exprs_of(h):
                is_src = isinstance(h, Assign) and e is h.src

                def visit(x, parent, chain=False):
                    # chain: every ancestor up to the root is a __paddw
                    if isinstance(x, Var) and x.var in C:
                        ok = is_src and acc_def and (x is e or chain)
                        ok = ok or (is_src and h_def and (parent is e or
                                    (isinstance(parent, Bin) and parent.op == ">>u" and parent is e.args[1])))
                        if not ok:
                            bad.add(x.var)
                    elif isinstance(x, Var) and x.var in H:
                        if not (isinstance(parent, Cast) and parent.ty == "i32"):
                            bad.add(hsteps_src[x.var])
                    for c in x.children():
                        visit(c, x, _is_paddw(x) and (x is e or chain))
                visit(e, None)
        return bad
    hsteps_src = {}
    C = set(cand)
    while True:
        changed = False
        for v in list(C):
            if not all(ok_def(d, C) for d in defs[v]):
                C.discard(v)
                changed = True
        H = {h for h, d in hsteps.items() if isinstance(_hstep(d.src), Var) and _hstep(d.src).var in C}
        hsteps_src = {h: _hstep(hsteps[h].src).var for h in H}
        bad = bad_uses(C, H) & C
        if bad:
            C -= bad
            changed = True
        if not changed:
            break
    if not C:
        return False
    H = {h for h, d in hsteps.items() if isinstance(_hstep(d.src), Var) and _hstep(d.src).var in C}
    # ---- rewrite: accumulators and h become int (sum of both lanes)
    for v in C | H:
        v.ty = "i32"

    def sc(e):
        """scalar (sum of lanes) of a __paddw tree with accumulators as leaves."""
        if isinstance(e, Var) and e.var in C:
            return Var(e.var, "i32")
        if _is_paddw(e):
            return Bin("+", sc(e.args[0]), sc(e.args[1]), "i32")
        return lanesum(e)
    for v in C:
        for d in defs[v]:
            s = d.src
            if isinstance(s, Const):
                d.src = Const(0, "i32")
            elif isinstance(s, Var):
                d.src = Var(s.var, "i32")
            else:
                d.src = simplify(sc(s))
            d.dst = Var(v, "i32")
    for hv in H:
        d = hsteps[hv]
        d.src = Var(_hstep(d.src).var, "i32")
        d.dst = Var(hv, "i32")

    def fix(e):
        def f(x):
            if isinstance(x, Cast) and x.ty == "i32" and isinstance(x.a, Var) and x.a.var in H:
                return Var(x.a.var, "i32")
            return x
        return simplify(map_expr(e, f))
    for h in roots:
        if isinstance(h, Assign) and isinstance(h.dst, Var) and h.dst.var in (C | H):
            continue
        _map_root(h, fix)
    return True


# ------------------------------------------------------------ packed float maps
#
# y[i] = a * x[i] + y[i] vectorised by two float lanes per 64-bit word:
#
#     t = __pfadds(__pfmuls(A, X[k+1]), Y[k+1]);  Y[k] = __pfadds(...);  Y[k+1] = t;
#
# becomes plain float code, lane by lane:
#
#     t_0 = a * x[2k+2] + y[2k+2];  t_1 = ...;  y[2k] = a * x[2k] + y[2k]; ...
#
# A variable is split into two float lanes when every definition is a packed
# float operation (or a constant) and every read is an operand of such an
# operation in a split definition / packed store, a whole 64-bit store, or an
# explicit lane extraction ((int)v, (int)(v >> 32), 32-bit store of v).  A
# 64-bit store becomes two 32-bit stores; the high lane is computed into a
# temporary first when its loads could read the low half just written.
def _is_pf(e):
    return isinstance(e, Intr) and e.name in PF and len(e.args) == 2


def _hi_of(x, C):
    """v >> 32 with v split -> v"""
    if isinstance(x, Bin) and x.op == ">>u" and isinstance(x.b, Const) and x.b.value == 32 \
            and isinstance(x.a, Var) and x.a.var in C:
        return x.a.var
    return None


def _may_overlap(load_addr, store_addr, width=4):
    d = simplify(Bin("-", load_addr, store_addr, "i64"))
    if not isinstance(d, Const):
        return True
    return -width < d.signed() < width


def _is_vop(e):
    return isinstance(e, Intr) and e.name in VOPS and len(e.args) == 2


def devectorize_maps(nodes, F):
    S = _S()
    if F is None:
        return False
    roots = _roots(nodes)
    defs = {}
    for h in roots:
        if isinstance(h, Assign) and isinstance(h.dst, Var):
            defs.setdefault(h.dst.var, []).append(h)
    C = {}
    for v, ds in defs.items():
        if v.ty != "i64" or any(getattr(d, "is_call", False) for d in ds):
            continue
        ops = [d.src for d in ds if _is_vop(d.src)]
        if not ops or not all(_is_vop(d.src) or isinstance(d.src, Const) for d in ds):
            continue
        fl = {o.name in PF for o in ops}
        if len(fl) == 1:
            C[v] = "f32" if fl.pop() else "i32"

    def vstore(h, C):
        return isinstance(h, Store) and h.ty == "i64" and \
            (_is_vop(h.val) or (isinstance(h.val, Var) and h.val.var in C))

    def bad_uses(C):
        bad = set()
        for h in roots:
            for e in _exprs_of(h):
                ctx = (isinstance(h, Assign) and e is h.src and isinstance(h.dst, Var) and
                       h.dst.var in C) or (vstore(h, C) and e is h.val)
                st32 = isinstance(h, Store) and h.ty == "i32" and e is h.val

                def visit(x, chain):
                    if isinstance(x, Cast) and x.ty == "i32" and \
                            ((isinstance(x.a, Var) and x.a.var in C) or _hi_of(x.a, C)):
                        return
                    if x is e and st32 and ((isinstance(x, Var) and x.var in C) or _hi_of(x, C)):
                        return
                    if isinstance(x, Var) and x.var in C:
                        if not (chain or x.ty == "i32"):
                            bad.add(x.var)
                        return
                    for c in x.children():
                        visit(c, chain and _is_vop(x))
                visit(e, ctx)
        return bad
    while C:
        bad = bad_uses(C) & set(C)
        if not bad:
            break
        for v in bad:
            del C[v]
    has_store = any(vstore(h, C) for h in roots)
    if not C and not has_store:
        return False
    from .analysis import VarInfo
    lanes = {}
    for v, ty in C.items():
        lanes[v] = []
        for k in (0, 1):
            nv = VarInfo("%s_%d" % (v.name, k), v.reg, ty)
            nv.fixed_name = nv.name
            nv.ast_def = True
            F.vars.append(nv)
            lanes[v].append(nv)
    ntemp = [0]

    def temp(ty):
        ntemp[0] += 1
        nv = VarInfo("vl%d" % ntemp[0], "-", ty)
        nv.fixed_name = nv.name
        nv.ast_def = True
        F.vars.append(nv)
        return nv

    def lane_of(e, k, ty):
        return simplify(lane_f(e, k, lanes) if ty == "f32" else lane(e, k, lanes))

    def lane_var_int(v, k):
        lv = lanes[v][k]
        return Var(lv, "i32") if lv.ty == "i32" else BitCast(Var(lv, lv.ty), "i32")

    def fix(e):
        """lane extractions of split variables"""
        def f(x):
            if isinstance(x, Cast) and x.ty == "i32":
                if isinstance(x.a, Var) and x.a.var in C:
                    return lane_var_int(x.a.var, 0)
                hv = _hi_of(x.a, C)
                if hv is not None:
                    return lane_var_int(hv, 1)
            if isinstance(x, Var) and x.var in C and x.ty == "i32":
                return lane_var_int(x.var, 0)
            return x
        return simplify(map_expr(e, f))

    def rewrite(ns):
        out = []
        for n in ns:
            if isinstance(n, S.SIf):
                n.then = rewrite(n.then)
                n.els = rewrite(n.els)
            elif isinstance(n, S.SLoop):
                n.body = rewrite(n.body)
            elif isinstance(n, S.SSwitch):
                n.cases = [(ks, rewrite(b)) for ks, b in n.cases]
                if n.default:
                    n.default = rewrite(n.default)
            if not isinstance(n, S.SStmt):
                out.append(n)
                continue
            st = n.s
            g = st.guard
            if isinstance(st, Assign) and isinstance(st.dst, Var) and st.dst.var in C:
                ty = C[st.dst.var]
                for k in (0, 1):
                    out.append(S.SStmt(Assign(Var(lanes[st.dst.var][k], ty), lane_of(st.src, k, ty), g)))
                continue
            if vstore(st, C):
                a = st.addr
                a4 = simplify(Bin("+", a, Const(4, "i64"), "i64"))
                lo, hi = lane_of(st.val, 0, "i32"), lane_of(st.val, 1, "i32")
                if any(isinstance(x, Load) and _may_overlap(x.addr, a) for x in hi.walk()):
                    hf = simplify(hi.a) if isinstance(hi, BitCast) else hi
                    t = temp(hf.ty if hf.ty == "f32" else "i32")
                    out.append(S.SStmt(Assign(Var(t, t.ty), hf, g)))
                    hi = BitCast(Var(t, "f32"), "i32") if t.ty == "f32" else Var(t, "i32")
                out.append(S.SStmt(Store(a, lo, "i32", g)))
                out.append(S.SStmt(Store(a4, hi, "i32", g)))
                continue
            if isinstance(st, Store) and st.ty == "i32" and C:
                v = st.val
                if isinstance(v, Var) and v.var in C:
                    st.val = lane_var_int(v.var, 0)
                elif _hi_of(v, C) is not None:
                    st.val = lane_var_int(_hi_of(v, C), 1)
            if C:
                st.map(fix)
            out.append(n)
        return out
    nodes[:] = rewrite(nodes)
    if C:
        for h in _roots(nodes):
            if not isinstance(h, (Assign, Store)):
                _map_root(h, fix)
    return True
