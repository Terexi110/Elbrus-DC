"""Data-flow analyses: call arguments, register webs -> variables, typing,
expression propagation, dead code elimination."""
from .ir import (Assign, Store, ExprStmt, Branch, Return, IJump, Jump, Reg, Var, Call,
                 Intr, Load, Const, FConst, Sym, Str, Cast, BitCast, Select, Un, Bin,
                 map_expr, is_float, is_int, BITS)
from .simplify import simplify


class VarInfo:
    def __init__(self, name, reg, ty="i64"):
        self.name, self.reg, self.ty = name, reg, ty
        self.is_param = False
        self.param_index = None
        self.is_input = False
        self.def_tys = []
        self.use_tys = []
        self.pointer = False
        self.def_srcs = []

    def __repr__(self):
        return "<Var %s:%s>" % (self.name, self.ty)


def all_stmts(F):
    for b in F.block_list():
        for i, s in enumerate(b.stmts):
            yield b, i, s


def stmt_exprs(s):
    return s.exprs()


# ------------------------------------------------------------ call arguments
def regs_written_before(F, blk, idx, limit=4):
    """Map reg->type written before position idx in blk (and single-pred chain)
    since the previous call."""
    out = {}
    b, i = blk, idx
    depth = 0
    while True:
        for s in reversed(b.stmts[:i]):
            if isinstance(s, Assign) and isinstance(s.src, Call):
                return out
            if isinstance(s, Assign) and isinstance(s.dst, Reg):
                out.setdefault(s.dst.name, s.dst.ty if s.dst.ty else s.src.ty)
        depth += 1
        if depth > limit or len(b.preds) != 1:
            return out
        b = F.blocks[b.preds[0]]
        i = len(b.stmts)


def assign_call_args(F, summaries, fparams=None):
    for blk, st in F.calls:
        call = st.src
        info = call.info
        idx = blk.stmts.index(st)
        base = 2 * info.wbs
        written = regs_written_before(F, blk, idx)
        n = 0
        tys = {}
        for k in range(8):
            r = "r%d" % (base + k)
            if r in written:
                n = k + 1
                tys[k] = written[r]
        tname = call.target.name if isinstance(call.target, Sym) else None
        known = summaries.get(tname) if tname else None
        if known is not None:
            n = known.get("nparams", n)
        if getattr(call, "tail", False) and known is None and fparams is not None:
            n = max(n, fparams)
        args = []
        for k in range(n):
            ty = tys.get(k, "i64")
            if ty == "b":
                ty = "i32"
            args.append(Reg("r%d" % (base + k), ty if ty in ("i32", "i64", "f32", "f64") else "i64"))
        call.args = args
        call.nfixed = n


def decide_void(F):
    """Heuristic: function returns a value iff some return block writes %r0."""
    rets = [b for b in F.block_list() if isinstance(b.term, Return)]
    has = False
    for b in rets:
        for s in b.stmts:
            if isinstance(s, Assign) and isinstance(s.dst, Reg) and s.dst.name == "r0":
                has = True
        synthetic = b.addr & 7 != 0          # taken side of a conditional return
        if len(b.stmts) == 0 and len(b.preds) == 1 and (synthetic or isinstance(F.blocks[b.preds[0]].term, Jump)):
            p = F.blocks[b.preds[0]]
            for s in p.stmts:
                if isinstance(s, Assign) and isinstance(s.dst, Reg) and s.dst.name == "r0":
                    has = True
    if not has:
        for b in rets:
            b.term = Return(None)
    F.returns_value = has
    return has


# ------------------------------------------------------------ webs
class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def term_exprs(t):
    return t.exprs() if t is not None else []


def build_vars(F):
    blocks = F.block_list()
    defs = []                       # (blk, idx, name, guarded)
    reg_defs = {}
    used = set()
    for b in blocks:
        for i, s in enumerate(b.stmts):
            for e in s.exprs():
                for x in e.walk():
                    if isinstance(x, Reg):
                        used.add(x.name)
            if isinstance(s, Assign) and isinstance(s.dst, Reg):
                d = len(defs)
                defs.append((b, i, s.dst.name, s.guard is not None))
                reg_defs.setdefault(s.dst.name, []).append(d)
                if s.guard is not None:
                    used.add(s.dst.name)
        for e in term_exprs(b.term):
            for x in e.walk():
                if isinstance(x, Reg):
                    used.add(x.name)
    # entry pseudo-defs
    entry_def = {}
    for r in sorted(used | set(reg_defs)):
        d = len(defs)
        defs.append((None, -1, r, False))
        entry_def[r] = d
        reg_defs.setdefault(r, []).append(d)
    regmask = {r: sum(1 << d for d in ds) for r, ds in reg_defs.items()}
    gen, kill = {}, {}
    for b in blocks:
        g, k = 0, 0
        for i, s in enumerate(b.stmts):
            if isinstance(s, Assign) and isinstance(s.dst, Reg):
                d = next(j for j in reg_defs[s.dst.name] if defs[j][0] is b and defs[j][1] == i)
                if s.guard is None:
                    g &= ~regmask[s.dst.name]
                    k |= regmask[s.dst.name]
                g |= 1 << d
        gen[b.addr], kill[b.addr] = g, k
    entry_in = sum(1 << d for d in entry_def.values())
    IN = {b.addr: 0 for b in blocks}
    OUT = {b.addr: gen[b.addr] for b in blocks}
    changed = True
    order = rpo(F)
    while changed:
        changed = False
        for a in order:
            b = F.blocks[a]
            i = entry_in if a == F.entry else 0
            for p in b.preds:
                i |= OUT[p]
            if a == F.entry:
                i |= entry_in
            o = gen[a] | (i & ~kill[a])
            if i != IN[a] or o != OUT[a]:
                IN[a], OUT[a] = i, o
                changed = True
    uf = UF()
    for d in range(len(defs)):
        uf.find(d)
    use_sites = []                  # (stmt-or-term, Reg obj, reaching defs list)
    def_index = {}
    for d, (b, i, r, g) in enumerate(defs):
        if b is not None:
            def_index[(b.addr, i)] = d

    def bits(x):
        out = []
        while x:
            low = x & -x
            out.append(low.bit_length() - 1)
            x ^= low
        return out

    reach_of_use = {}
    pure_entry = set()
    for b in blocks:
        cur = IN.get(b.addr, 0)
        if b.addr not in IN:
            cur = 0
        for i, s in enumerate(b.stmts):
            for e in s.exprs():
                for x in e.walk():
                    if isinstance(x, Reg):
                        rd = bits(cur & regmask.get(x.name, 0)) or [entry_def[x.name]]
                        if entry_def[x.name] in rd and any(
                                not defs[d][3] for d in rd if d != entry_def[x.name]) \
                                or rd == [entry_def[x.name]]:
                            pure_entry.add(x.name)
                        for d in rd[1:]:
                            uf.union(rd[0], d)
                        reach_of_use[id(x)] = rd[0]
            if isinstance(s, Assign) and isinstance(s.dst, Reg):
                d = def_index[(b.addr, i)]
                if s.guard is not None:
                    rd = bits(cur & regmask[s.dst.name]) or [entry_def[s.dst.name]]
                    for x in rd:
                        uf.union(d, x)
                    cur |= 1 << d
                else:
                    cur &= ~regmask[s.dst.name]
                    cur |= 1 << d
        for e in term_exprs(b.term):
            for x in e.walk():
                if isinstance(x, Reg):
                    rd = bits(cur & regmask.get(x.name, 0)) or [entry_def[x.name]]
                    if entry_def[x.name] in rd and any(
                            not defs[d][3] for d in rd if d != entry_def[x.name]) \
                            or rd == [entry_def[x.name]]:
                        pure_entry.add(x.name)
                    for d in rd[1:]:
                        uf.union(rd[0], d)
                    reach_of_use[id(x)] = rd[0]
    # ---- variables per web
    var_of_root = {}
    counter = [0]
    F.vars = []

    def var_for(d):
        root = uf.find(d)
        v = var_of_root.get(root)
        if v is None:
            members = [m for m in range(len(defs)) if uf.find(m) == root]
            reg = defs[d][2]
            is_entry = any(defs[m][0] is None for m in members)
            has_real = any(defs[m][0] is not None for m in members)
            v = VarInfo("", reg)
            if is_entry and (not has_real or reg in pure_entry or reg == "sp"):
                v.is_input = True
                if reg.startswith("r") and reg[1:].isdigit() and int(reg[1:]) < 8:
                    v.is_param = True
                    v.param_index = int(reg[1:])
            v.has_defs = has_real
            var_of_root[root] = v
            F.vars.append(v)
        return v

    def sub_use(x):
        if isinstance(x, Reg):
            d = reach_of_use.get(id(x))
            if d is None:
                d = entry_def[x.name]
            v = var_for(d)
            if x.ty != "any":
                v.use_tys.append(x.ty)
            return Var(v, x.ty)
        return x

    def rw(e):
        if isinstance(e, Reg):
            return sub_use(e)
        return e.replace(rw)

    for b in blocks:
        for i, s in enumerate(b.stmts):
            if isinstance(s, Assign):
                s.src = rw(s.src)
                if s.guard is not None:
                    s.guard = rw(s.guard)
                if isinstance(s.dst, Reg):
                    d = def_index[(b.addr, i)]
                    v = var_for(d)
                    ty = s.dst.ty
                    v.def_tys.append(s.src.ty if s.src.ty else ty)
                    v.def_srcs.append(s.src)
                    s.dst = Var(v, ty)
            else:
                s.map(rw)
        if b.term is not None:
            b.term.map(rw)
    assign_types(F)
    name_vars(F)


def assign_types(F):
    for v in F.vars:
        tys = [t for t in v.def_tys if t and t != "any"] or []
        if not tys and v.def_srcs and all(isinstance(x, Var) and x.ty == "any" for x in v.def_srcs):
            v.copy_only = True
        uses = [t for t in v.use_tys if t]
        cand = tys or uses or ["i64"]
        if "b" in cand:
            v.ty = "b"
        elif any(is_float(t) for t in cand):
            fl = [t for t in cand if is_float(t)]
            v.ty = max(set(fl), key=fl.count)
        elif v.is_input and not tys:
            v.ty = "i32" if uses and all(t == "i32" for t in uses) else "i64"
        else:
            ints = [t for t in cand if is_int(t)]
            narrow_uses = uses and all(t in ("i32", "i16", "i8") for t in uses)
            narrow_defs = all(narrow_src(x) for x in v.def_srcs) if v.def_srcs else False
            if ints and narrow_uses and narrow_defs:
                v.ty = "i32"
            elif ints and all(t == "i32" for t in ints):
                # 32-bit defs; if every use is 64-bit it is still an int
                v.ty = "i32"
            elif ints:
                v.ty = "i64"
            else:
                v.ty = "i64"
    # copies take the type of their source (fixpoint)
    for _ in range(10):
        ch = False
        for v in F.vars:
            if getattr(v, "copy_only", False):
                srcs = {x.var.ty for x in v.def_srcs if isinstance(x, Var)}
                srcs.discard(None)
                if len(srcs) == 1:
                    t = next(iter(srcs))
                    if t != v.ty:
                        v.ty = t
                        ch = True
        if not ch:
            break

    # rewrite reads/writes with conversions
    def conv_read(e):
        if isinstance(e, Var) and e.ty == "any":
            return Var(e.var, e.var.ty)
        if isinstance(e, Var) and e.ty != e.var.ty:
            want, have = e.ty, e.var.ty
            base = Var(e.var, have)
            return convert(base, want)
        return e.replace(conv_read)

    for b in F.block_list():
        for s in b.stmts:
            if isinstance(s, Assign):
                s.src = conv_read(s.src)
                if s.guard is not None:
                    s.guard = conv_read(s.guard)
                if isinstance(s.dst, Var):
                    v = s.dst.var
                    if s.src.ty != v.ty:
                        s.src = convert(s.src, v.ty)
                    s.dst = Var(v, v.ty)
            else:
                s.map(conv_read)
        if b.term is not None:
            b.term.map(conv_read)


def narrow_src(e):
    if e.ty in ("i32", "i16", "i8"):
        return True
    if isinstance(e, Cast) and is_int(e.a.ty) and BITS[e.a.ty] <= 32:
        return True
    if isinstance(e, Const) and -2**31 <= (e.value - (1 << 64) if e.value >> 63 else e.value) < 2**31:
        return True
    return False


def convert(e, want):
    have = e.ty
    if have == want:
        return e
    if have == "b" and is_int(want):
        return Cast(e, want, signed=False)
    if want == "b":
        return Bin("!=", e, Const(0, have if is_int(have) else "i64"), "b")
    if is_float(want) != is_float(have):
        if BITS.get(want) == BITS.get(have):
            return BitCast(e, want)
        if is_float(want):
            mid = "i%d" % BITS[want]
            return BitCast(Cast(e, mid, signed=False), want)
        mid = "i%d" % BITS[have]
        return Cast(BitCast(e, mid), want, signed=False)
    if is_float(want):
        return Cast(e, want)
    if BITS.get(want, 64) < BITS.get(have, 64):
        return Cast(e, want, signed=True)
    return Cast(e, want, signed=False)


def name_vars(F):
    n = 0
    for v in F.vars:
        if v.is_param:
            v.name = "a%d" % (v.param_index + 1)
        elif v.reg == "sp":
            v.name = "sp"
        elif v.is_input:
            v.name = "in_" + v.reg
        else:
            n += 1
            v.name = ("p%d" if v.ty == "b" else "v%d") % n


# ------------------------------------------------------------ CFG helpers
def rpo(F):
    seen, out = set(), []

    def dfs(a):
        stack = [(a, iter(F.blocks[a].succs))]
        seen.add(a)
        while stack:
            node, it = stack[-1]
            for s in it:
                if s not in seen and s in F.blocks:
                    seen.add(s)
                    stack.append((s, iter(F.blocks[s].succs)))
                    break
            else:
                stack.pop()
                out.append(node)
    if F.entry in F.blocks:
        dfs(F.entry)
    for a in F.order:
        if a not in seen:
            dfs(a)
    return list(reversed(out))


def dominators(F):
    order = rpo(F)
    idx = {a: i for i, a in enumerate(order)}
    idom = {F.entry: F.entry}

    def intersect(a, b):
        while a != b:
            while idx[a] > idx[b]:
                a = idom[a]
            while idx[b] > idx[a]:
                b = idom[b]
        return a
    changed = True
    while changed:
        changed = False
        for a in order:
            if a == F.entry:
                continue
            ps = [p for p in F.blocks[a].preds if p in idom]
            if not ps:
                continue
            new = ps[0]
            for p in ps[1:]:
                new = intersect(p, new)
            if idom.get(a) != new:
                idom[a] = new
                changed = True
    return idom


def dominates(idom, a, b):
    while True:
        if a == b:
            return True
        nb = idom.get(b)
        if nb is None or nb == b:
            return a == b
        b = nb


# ------------------------------------------------------------ propagation
def vars_in(e):
    return {x.var for x in e.walk() if isinstance(x, Var)}


def is_memory_stmt(s):
    if isinstance(s, Store):
        return True
    for e in s.exprs():
        for x in e.walk():
            if isinstance(x, Call) or (isinstance(x, Intr) and not x.pure):
                return True
    return False


def has_load(e):
    return any(isinstance(x, (Load, Call)) or (isinstance(x, Intr) and not x.pure) for x in e.walk())


def has_call(e):
    return any(isinstance(x, Call) or (isinstance(x, Intr) and not x.pure) for x in e.walk())


def stmt_defvar(s):
    if isinstance(s, Assign) and isinstance(s.dst, Var):
        return s.dst.var
    return None


def collect(F):
    defs, uses = {}, {}
    for b in F.block_list():
        for i, s in enumerate(b.stmts):
            v = stmt_defvar(s)
            if v is not None:
                defs.setdefault(v, []).append((b, i))
                if s.guard is not None:
                    uses.setdefault(v, []).append((b, i))
            for e in s.exprs():
                for x in e.walk():
                    if isinstance(x, Var):
                        uses.setdefault(x.var, []).append((b, i))
        for e in term_exprs(b.term):
            for x in e.walk():
                if isinstance(x, Var):
                    uses.setdefault(x.var, []).append((b, len(b.stmts)))
    return defs, uses


def subst(e, var, rep):
    def f(x):
        if isinstance(x, Var) and x.var is var:
            return rep
        return x
    return map_expr(e, f)


def blocks_between(F, src, dst):
    """Blocks on some path src -> ... -> dst (excluding src/dst endpoints themselves)."""
    fwd = set()
    stack = list(F.blocks[src].succs)
    while stack:
        a = stack.pop()
        if a in fwd or a == src:
            continue
        fwd.add(a)
        stack.extend(F.blocks[a].succs)
    bwd = set()
    stack = list(F.blocks[dst].preds)
    while stack:
        a = stack.pop()
        if a in bwd or a == dst:
            continue
        bwd.add(a)
        stack.extend(F.blocks[a].preds)
    return (fwd & bwd), (dst in fwd), (src in bwd)


def can_move(F, defs, db, di, ub, ui, e, idom):
    evars = vars_in(e)
    mem = has_load(e)
    call = has_call(e)

    def check_range(b, lo, hi):
        for k in range(lo, hi):
            s = b.stmts[k]
            v = stmt_defvar(s)
            if v is not None and v in evars:
                return False
            if (mem and is_memory_stmt(s)) or (call and any(has_load(x) for x in s.exprs())):
                return False
        return True

    if db is ub:
        if ui <= di:
            return False
        return check_range(db, di + 1, ui)
    if mem:
        return False
    if not dominates(idom, db.addr, ub.addr):
        return False
    mid, _, loop_back = blocks_between(F, db.addr, ub.addr)
    if ub.addr in mid or db.addr in mid:
        return False
    for v in evars:
        for (b, i) in defs.get(v, []):
            if b.addr in mid:
                return False
            if b is db and i > di:
                return False
            if b is ub and i < ui:
                return False
    return True


def cheap(e):
    return isinstance(e, (Const, FConst, Sym, Str)) or (isinstance(e, Var))


def complementary(g1, g2):
    return (isinstance(g2, Un) and g2.op == "!" and g2.a == g1) or \
           (isinstance(g1, Un) and g1.op == "!" and g1.a == g2)


def merge_guarded_pairs(F):
    """if (g) v = A; if (!g) v = B;  ->  v = g ? A : B  (B must not read v)"""
    for b in F.block_list():
        i = 0
        while i + 1 < len(b.stmts):
            s1 = b.stmts[i]
            j = i + 1
            found = None
            while j < len(b.stmts) and j <= i + 6:
                s2 = b.stmts[j]
                if isinstance(s1, Assign) and isinstance(s2, Assign) and isinstance(s1.dst, Var) \
                        and isinstance(s2.dst, Var) and s1.dst.var is s2.dst.var \
                        and s1.guard is not None and s2.guard is not None \
                        and complementary(s1.guard, s2.guard):
                    found = j
                    break
                # stop on anything touching the same var or guard vars
                if isinstance(s1, Assign) and isinstance(s1.dst, Var):
                    v = s1.dst.var
                    touched = stmt_defvar(s2) is v or any(v in vars_in(e) for e in s2.exprs())
                    gv = vars_in(s1.guard) if s1.guard is not None else set()
                    if touched or (stmt_defvar(s2) in gv):
                        break
                j += 1
            if found is not None:
                s2 = b.stmts[found]
                v = s1.dst.var
                if v not in vars_in(s2.src):
                    s2.src = Select(s1.guard, s1.src, s2.src, s2.src.ty)
                    s2.guard = None
                    del b.stmts[i]
                    continue
            i += 1


def merge_selects(F):
    """v = a; if (g) v = b;  ->  v = g ? b : a"""
    merge_guarded_pairs(F)
    changed = True
    while changed:
        changed = False
        defs, uses = collect(F)
        for v, ds in defs.items():
            if len(ds) != 2:
                continue
            (b1, i1), (b2, i2) = ds
            if b1 is not b2 or i2 <= i1:
                continue
            s1, s2 = b1.stmts[i1], b1.stmts[i2]
            if s2.guard is None:
                continue
            if v in vars_in(s2.src) or v in vars_in(s2.guard):
                continue
            # no use of v between
            if any(u for u in uses.get(v, []) if u[0] is b1 and i1 < u[1] < i2):
                continue
            if s1.guard is not None:
                g1, g2 = s1.guard, s2.guard
                comp = (isinstance(g2, Un) and g2.op == "!" and g2.a == g1) or \
                       (isinstance(g1, Un) and g1.op == "!" and g1.a == g2)
                if not comp:
                    continue
                if not can_move(F, defs, b1, i1, b1, i2, s1.src, None):
                    continue
                s2.src = Select(s2.guard, s2.src, s1.src, s2.src.ty)
                s2.guard = None
                del b1.stmts[i1]
                changed = True
                break
            if not can_move(F, defs, b1, i1, b1, i2, s1.src, None):
                continue
            if v in vars_in(s1.src):
                continue
            s2.src = Select(s2.guard, s2.src, s1.src, s2.src.ty)
            s2.guard = None
            del b1.stmts[i1]
            changed = True
            break


def propagate(F, max_rounds=2000):
    idom = dominators(F)
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        defs, uses = collect(F)
        done = False
        for v in list(defs):
            ds = defs[v]
            if len(ds) != 1 or v.is_input:
                continue
            db, di = ds[0]
            s = db.stmts[di]
            if s.guard is not None:
                continue
            us = uses.get(v, [])
            e = s.src
            if not us:
                if has_call(e):
                    db.stmts[di] = ExprStmt(e, None, getattr(s, "addr", None))
                    if isinstance(e, Call):
                        e.ret_used = False
                else:
                    del db.stmts[di]
                done = True
                break
            sp_rel = isinstance(e, Bin) and e.op == "+" and isinstance(e.a, Var) and e.a.var.reg == "sp" \
                and isinstance(e.b, Const)
            if isinstance(e, (Const, FConst, Sym, Str)) or sp_rel:
                ok = True
            elif isinstance(e, Var):
                ok = all(can_move(F, defs, db, di, ub, ui, e, idom) for ub, ui in us)
            else:
                if len(us) != 1:
                    continue
                ub, ui = us[0]
                if has_call(e):
                    tgt = ub.stmts[ui] if ui < len(ub.stmts) else ub.term
                    if getattr(tgt, "guard", None) is not None:
                        continue
                ok = can_move(F, defs, db, di, ub, ui, e, idom)
            if not ok:
                continue
            for ub, ui in set((id(b), i) for b, i in us) and us:
                pass
            seen = set()
            for ub, ui in us:
                if (ub.addr, ui) in seen:
                    continue
                seen.add((ub.addr, ui))
                if ui < len(ub.stmts):
                    ub.stmts[ui].map(lambda x: subst(x, v, e))
                else:
                    ub.term.map(lambda x: subst(x, v, e))
            del db.stmts[di]
            done = True
            break
        if not done:
            break
    simplify_all(F)


def dce(F):
    changed = True
    while changed:
        changed = False
        defs, uses = collect(F)
        for v, ds in defs.items():
            if v.is_input or uses.get(v):
                continue
            # all defs unused: remove pure ones
            for (b, i) in sorted(ds, key=lambda t: -t[1]):
                s = b.stmts[i]
                if has_call(s.src):
                    b.stmts[i] = ExprStmt(s.src, s.guard, getattr(s, "addr", None))
                    if isinstance(s.src, Call):
                        s.src.ret_used = False
                else:
                    del b.stmts[i]
                changed = True
            if changed:
                break


def simplify_all(F):
    for b in F.block_list():
        for s in b.stmts:
            s.map(simplify)
        if b.term is not None:
            b.term.map(simplify)
