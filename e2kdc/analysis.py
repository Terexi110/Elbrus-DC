"""Data-flow analyses: call arguments, register webs -> variables, typing,
expression propagation, dead code elimination."""
from .ir import (Assign, Store, ExprStmt, Branch, Return, IJump, Jump, Reg, Var, Call,
                 Intr, Load, Const, FConst, Sym, Str, Cast, BitCast, Select, Un, Bin,
                 map_expr, is_float, is_int, BITS)
from .simplify import simplify
from . import libc


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
        call.known_arity = False
        if known is not None:
            n = known.get("nparams", n)
            call.known_arity = "nparams" in known
        elif tname and libc.lookup(tname):
            # a C library function: its fixed parameters are always passed; a
            # variadic one may take more (counted from the format later)
            decl, _ret, params, fmt = libc.lookup(tname)
            n = max(n, len(params)) if "..." in decl else len(params)
            call.known_arity = "..." not in decl
            for k in range(len(params)):
                if k not in tys:
                    tys[k] = "i64" if params[k] not in ("int",) else "i32"
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
    site_def = {(id(b), i): d for d, (b, i, _, _) in enumerate(defs) if b is not None}
    for b in blocks:
        g, k = 0, 0
        for i, s in enumerate(b.stmts):
            if isinstance(s, Assign) and isinstance(s.dst, Reg):
                d = site_def[(id(b), i)]
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
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
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

    members_of = {}                 # web root -> its definitions, ascending
    for m in range(len(defs)):
        members_of.setdefault(uf.find(m), []).append(m)

    def var_for(d):
        root = uf.find(d)
        v = var_of_root.get(root)
        if v is None:
            members = members_of[root]
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
            wide = max((BITS.get(t, 0) for t in tys), default=0)
            if wide > BITS[v.ty]:
                v.ty = "i%d" % wide         # a 64-bit def (packed pair of floats) shares the web
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
        if getattr(v, "fixed_name", None):
            v.name = v.fixed_name
        elif v.is_param:
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
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
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


def _var_reads(e, out):
    """append the VarInfo of every Var node of e to out (with repetitions)"""
    stack = [e]
    pop, push = stack.pop, stack.extend
    while stack:
        x = pop()
        if type(x) is Var:
            out.append(x.var)
        else:
            ch = x.children()
            if ch:
                push(ch[::-1])


def _scan_block(b, defs, uses, seen=None):
    """add the definitions and reads of block b to defs / uses (collect's
    order within the block); seen: set filled with every variable met"""
    stmts = b.stmts
    reads = []
    for i, s in enumerate(stmts):
        if isinstance(s, Assign) and type(s.dst) is Var:
            v = s.dst.var
            defs.setdefault(v, []).append((b, i))
            if seen is not None:
                seen.add(v)
            if s.guard is not None:
                uses.setdefault(v, []).append((b, i))
        for e in s.exprs():
            _var_reads(e, reads)
        if reads:
            p = (b, i)
            for v in reads:
                uses.setdefault(v, []).append(p)
            if seen is not None:
                seen.update(reads)
            reads.clear()
    if b.term is not None:
        for e in b.term.exprs():
            _var_reads(e, reads)
        if reads:
            p = (b, len(stmts))
            for v in reads:
                uses.setdefault(v, []).append(p)
            if seen is not None:
                seen.update(reads)


def collect(F):
    defs, uses = {}, {}
    for b in F.block_list():
        _scan_block(b, defs, uses)
    return defs, uses


class DefUse:
    """collect(F) kept up to date by rescanning only the blocks a rewrite
    touched.  defs keeps collect's key order (first definition in block
    order: a rewrite never adds a definition, so a rescan only drops keys);
    the order of positions inside a list may differ from collect's, and no
    user depends on it except for len() / membership / two defs of one block
    (which a rescan appends in statement order)."""

    def __init__(self, F):
        self.F = F
        self.defs, self.uses = {}, {}
        self.bvars = {}
        self.ver = {}            # id(block) -> number of rescans (for caches of verdicts)
        for b in F.block_list():
            vs = set()
            _scan_block(b, self.defs, self.uses, vs)
            self.bvars[id(b)] = vs

    def rescan(self, blocks):
        blocks = list({id(b): b for b in blocks}.values())
        ids = {id(b) for b in blocks}
        ver = self.ver
        for k in ids:
            ver[k] = ver.get(k, 0) + 1
        touched = set()
        for b in blocks:
            touched |= self.bvars[id(b)]
        defs, uses = self.defs, self.uses
        for v in touched:
            for m in (defs, uses):
                lst = m.get(v)
                if lst:
                    lst[:] = [p for p in lst if id(p[0]) not in ids]
        for b in blocks:
            vs = set()
            _scan_block(b, defs, uses, vs)
            self.bvars[id(b)] = vs
        for b in blocks:
            touched |= self.bvars[id(b)]
        for v in touched:
            if v in defs and not defs[v]:
                del defs[v]
            if v in uses and not uses[v]:
                del uses[v]
        return touched


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


def can_move(F, defs, db, di, ub, ui, e, idom, bb=None):
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
    if bb is None:
        mid, _, loop_back = blocks_between(F, db.addr, ub.addr)
    else:                   # cache of a pass that keeps the CFG
        k = (db.addr, ub.addr)
        r = bb.get(k)
        if r is None:
            r = bb[k] = blocks_between(F, db.addr, ub.addr)
        mid, _, loop_back = r
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
    _rounds = 0
    du = DefUse(F)
    defs, uses = du.defs, du.uses
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
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
                du.rescan([b1])
                changed = True
                break
            if not can_move(F, defs, b1, i1, b1, i2, s1.src, None):
                continue
            if v in vars_in(s1.src):
                continue
            s2.src = Select(s2.guard, s2.src, s1.src, s2.src.ty)
            s2.guard = None
            del b1.stmts[i1]
            du.rescan([b1])
            changed = True
            break


def propagate(F, max_rounds=2000):
    """Copy / expression propagation: a variable defined once and unguarded
    is substituted into its uses (when can_move allows it) and its definition
    removed, one variable at a time, always the first suitable one in
    collect's order of definitions.

    Worklist form of "rescan everything, take the first that fits, repeat":
    a variable refused once stays refused until a block it depends on is
    rewritten (its definition's, its uses', the definitions' of the
    variables it reads; the CFG is fixed here) -- every such rewrite goes
    through DefUse.rescan, which names the variables it touched.  The heap
    holds the variables not known to be refused, by their place in that
    order, so the choice is the same as rescanning from the start."""
    import heapq
    idom = dominators(F)
    du = DefUse(F)
    defs, uses = du.defs, du.uses
    bb = {}
    pos = {v: k for k, v in enumerate(defs)}
    heap = list(enumerate(defs))       # (pos, VarInfo): already a heap
    queued = set(defs)
    dep_of = {}                         # id(block) -> vars refused by looking at it

    def refuse(v, db, us, e):
        ks = {id(db)}
        for b, _ in us:
            ks.add(id(b))
        for x in vars_in(e):
            for b, _ in defs.get(x, ()):
                ks.add(id(b))
        for k in ks:
            dep_of.setdefault(k, set()).add(v)

    def rewritten(blocks):
        touched = du.rescan(blocks)
        for b in blocks:
            touched |= dep_of.pop(id(b), set())
        for x in touched:
            if x not in queued and x in pos:
                queued.add(x)
                heapq.heappush(heap, (pos[x], x))

    rounds = 0
    while rounds < max_rounds and heap:
        _, v = heapq.heappop(heap)
        queued.discard(v)
        ds = defs.get(v)
        if ds is None or len(ds) != 1 or v.is_input:
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
            rounds += 1
            rewritten([db])
            continue
        sp_rel = isinstance(e, Bin) and e.op == "+" and isinstance(e.a, Var) and e.a.var.reg == "sp" \
            and isinstance(e.b, Const)
        if isinstance(e, (Const, FConst, Sym, Str)) or sp_rel:
            ok = True
        elif isinstance(e, Var):
            ok = all(can_move(F, defs, db, di, ub, ui, e, idom, bb) for ub, ui in us)
        else:
            if len(us) != 1:
                # every use sets a loop counter (lc / pc): substitute so the LSR
                # fields fold  (lc = (x & 0xffffffff) + (x >> 32 & 31), pc = x >> 48 & 31)
                def lsr_use(b, i):
                    t = b.stmts[i] if i < len(b.stmts) else None
                    return isinstance(t, Assign) and isinstance(t.dst, Var) and \
                        t.dst.var.reg in ("lc", "pc", "lsr_vlc", "lsr_strmd", "lsr_semc") and t.guard is None
                if not (all(lsr_use(b, i) for b, i in us) and not has_load(e) and not has_call(e)
                        and all(can_move(F, defs, db, di, b, i, e, idom, bb) for b, i in us)):
                    refuse(v, db, us, e)
                    continue
                seen2 = set()
                for b, i in us:
                    if (b.addr, i) not in seen2:
                        seen2.add((b.addr, i))
                        b.stmts[i].map(lambda x: subst(x, v, e))
                del db.stmts[di]
                rounds += 1
                rewritten([db] + [b for b, i in us])
                continue
            ub, ui = us[0]
            if has_call(e):
                tgt = ub.stmts[ui] if ui < len(ub.stmts) else ub.term
                if getattr(tgt, "guard", None) is not None:
                    refuse(v, db, us, e)
                    continue
            ok = can_move(F, defs, db, di, ub, ui, e, idom, bb)
        if not ok:
            refuse(v, db, us, e)
            continue
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
        rounds += 1
        rewritten([db] + [b for b, i in us])
    simplify_all(F)


def dce(F):
    changed = True
    _rounds = 0
    du = DefUse(F)
    defs, uses = du.defs, du.uses
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
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
                du.rescan([b for b, i in ds])
                break


def dce_live(F):
    """Dead-assignment elimination with *strong* liveness (faint variables):
    a read counts only if the statement doing it is live itself.  This is the
    least fixpoint, so self-sustaining copy cycles (the %b[] / %pred rotation
    model) die too, per definition, not per variable."""
    def vs(e):
        return {x.var for x in e.walk() if isinstance(x, Var)}

    def effect(s):
        return not (isinstance(s, Assign) and isinstance(s.dst, Var)) or has_call(s.src) \
            or getattr(s, "is_call", False)

    def transfer(b, live):
        live = set(live)
        marks = []
        for s in reversed(b.stmts):
            if effect(s):
                marks.append(True)
                for e in s.exprs():
                    live |= vs(e)
                if isinstance(s, Assign) and not isinstance(s.dst, Var):
                    live |= vs(s.dst)
                v = stmt_defvar(s)
                if v is not None and s.guard is None:
                    live.discard(v)
                    live |= vs(s.src) | (vs(s.guard) if s.guard is not None else set())
                continue
            v = s.dst.var
            if v in live:
                marks.append(True)
                if s.guard is None:
                    live.discard(v)
                live |= vs(s.src)
                if s.guard is not None:
                    live |= vs(s.guard)
            else:
                marks.append(False)
        marks.reverse()
        return live, marks

    blocks = F.block_list()
    live_in = {b.addr: set() for b in blocks}
    changed = True
    rounds = 0
    while changed and rounds < 200:
        changed = False
        rounds += 1
        for b in reversed(blocks):
            out = set()
            for sa in b.succs:
                out |= live_in.get(sa, set())
            for e in term_exprs(b.term):
                out |= vs(e)
            li, _ = transfer(b, out)
            if li != live_in[b.addr]:
                live_in[b.addr] = li
                changed = True
    removed = False
    for b in blocks:
        out = set()
        for sa in b.succs:
            out |= live_in.get(sa, set())
        for e in term_exprs(b.term):
            out |= vs(e)
        _, marks = transfer(b, out)
        keep = [s for s, m in zip(b.stmts, marks) if m]
        if len(keep) != len(b.stmts):
            removed = True
            b.stmts = keep
    return removed


def const_intr_args(F, all_vars=False):
    """Constant propagation by reaching definitions, applied only to arguments
    of intrinsics (getfd/insfd field specs, pshufb controls...): there a constant
    lets simplify() turn the intrinsic into plain C.  all_vars: every read of a
    variable whose reaching definitions all give the same constant (webs merged
    by a shared register, e.g. an offset that is 0 on the path into a kernel).  Lattice per variable:
    missing = no information yet, Const = that value on every path, None = varies."""
    from .ir import Intr
    blocks = F.block_list()
    NAC = None

    def meet(maps):
        out = {}
        keys = set()
        for m in maps:
            keys |= set(m)
        for k in keys:
            vals = [m.get(k, NAC) for m in maps]
            v0 = vals[0]
            out[k] = v0 if v0 is not NAC and all(v is not NAC and v.key() == v0.key() for v in vals) else NAC
        return out

    nodef = set()
    if all_vars:
        dd, _ = collect(F)
        nodef = {v for v in F.vars if not dd.get(v)}

    def stable(r):
        """an expression over never-assigned inputs (e.g. a broadcast of a param)"""
        return all(not isinstance(x, (Load, Call, Intr)) and
                   (not isinstance(x, Var) or x.var in nodef) for x in r.walk())

    def val(e, env):
        def f(x):
            if isinstance(x, Var) and env.get(x.var) is not NAC and x.var in env:
                return env[x.var]
            return x
        r = simplify(map_expr(e, f))
        if isinstance(r, Const) or (all_vars and not isinstance(r, Var) and stable(r)):
            return r
        return NAC

    def step(st, env):
        v = stmt_defvar(st)
        if v is not None:
            if st.guard is None and not has_load(st.src) and not has_call(st.src):
                env[v] = val(st.src, env)
            else:
                g = val(st.guard, env) if st.guard is not None else NAC
                if isinstance(g, Const) and not g.value and v in env:
                    pass                          # guard known false: value kept
                else:
                    env[v] = NAC
        return env
    inp = {b.addr: None for b in blocks}
    entry_env = {}
    out = {}
    changed = True
    rounds = 0
    while changed and rounds < 100:
        changed = False
        rounds += 1
        for b in blocks:
            if b.addr == F.entry:
                env = dict(entry_env)
            else:
                ps = [out[p] for p in b.preds if p in out]
                if not ps:
                    continue
                env = meet(ps)
            for st in b.stmts:
                step(st, env)
            if out.get(b.addr) != env:
                out[b.addr] = env
                changed = True
    # rewrite intrinsic arguments
    for b in blocks:
        if b.addr == F.entry:
            env = dict(entry_env)
        else:
            ps = [out[p] for p in b.preds if p in out]
            env = meet(ps) if ps else {}

        def fix(e):
            if isinstance(e, Var) and e.var.reg == "pc" and env.get(e.var) is not NAC and e.var in env:
                return env[e.var]                 # loop prologue counter known (usually 0)
            if all_vars and isinstance(e, Var) and env.get(e.var) is not NAC and e.var in env:
                c = env[e.var]
                if isinstance(c, Const) and (c.ty == e.ty or (is_int(c.ty) and is_int(e.ty))):
                    return Const(c.value, e.ty)
                x = c
                while isinstance(x, Cast) or (isinstance(x, Bin) and isinstance(x.b, Const)):
                    x = x.a
                if c.ty == e.ty and isinstance(x, Var):
                    return c                      # an alias of a param: (unsigned char)a2
                return e
            if isinstance(e, Intr):
                args = []
                for x in e.args:
                    if isinstance(x, Var) and env.get(x.var) is not NAC and x.var in env \
                            and env[x.var].ty == x.ty:
                        x = env[x.var]
                    args.append(x)
                e.args = args
            return e
        for st in b.stmts:
            st.map(lambda e: map_expr(e, fix))
            step(st, env)
        if all_vars and b.term is not None and hasattr(b.term, "map"):
            b.term.map(lambda e: map_expr(e, fix))


def live_at_entry(F):
    """Variables whose entry value can be read (plain backward liveness;
    a guarded definition does not kill)."""
    def vs(e):
        return {x.var for x in e.walk() if isinstance(x, Var)}
    blocks = F.block_list()
    live_in = {b.addr: set() for b in blocks}
    changed = True
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
        for b in reversed(blocks):
            live = set()
            for sa in b.succs:
                live |= live_in.get(sa, set())
            for e in term_exprs(b.term):
                live |= vs(e)
            for st in reversed(b.stmts):
                v = stmt_defvar(st)
                if v is not None and st.guard is None:
                    live.discard(v)
                for e in st.exprs():
                    live |= vs(e)
                if isinstance(st, Assign) and not isinstance(st.dst, Var):
                    live |= vs(st.dst)
            if live != live_in[b.addr]:
                live_in[b.addr] = live
                changed = True
    return live_in.get(F.entry, set())


def split_webs(F):
    """Re-split variables into webs after optimisation.  A register reused for
    unrelated values (%r0: first parameter, then a pointer, then the return
    value) keeps one variable from build_vars; after propagation/DCE the live
    ranges are often disjoint.  Definitions that reach a common use (a guarded
    definition counts as a use of the previous value) stay together; every other
    group gets its own variable.  The group holding the entry value keeps the
    original (parameter) variable."""
    import copy
    blocks = F.block_list()
    defs = []                    # did -> (var, stmt or None)
    defs_of = {}
    site = {}                    # id(stmt) -> did
    for b in blocks:
        for st in b.stmts:
            v = stmt_defvar(st)
            if v is not None and v.reg != "sp":
                site[id(st)] = len(defs)
                defs_of.setdefault(v, []).append(len(defs))
                defs.append((v, st))
    entry = {}
    for v in F.vars:
        if v.reg != "sp" and (v.is_input or v.is_param) and v in defs_of:
            entry[v] = len(defs)
            defs_of[v].append(len(defs))
            defs.append((v, None))
    # a dict, not a set: iterated below, and VarInfo hashes by address
    cand = {v: True for v, ds in defs_of.items() if len(ds) > 1}
    if not cand:
        return False
    parent = list(range(len(defs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def walk_block(b, env, visit):
        """env: var -> frozenset(dids) before the block; visit(var, dids, x, st)"""
        for st in b.stmts:
            for e in st.exprs():
                for x in e.walk():
                    if isinstance(x, Var) and x.var in cand:
                        visit(x.var, env.get(x.var, frozenset()))
            if isinstance(st, Assign) and not isinstance(st.dst, Var):
                for x in st.dst.walk():
                    if isinstance(x, Var) and x.var in cand:
                        visit(x.var, env.get(x.var, frozenset()))
            v = stmt_defvar(st)
            if v in cand:
                did = site[id(st)]
                if st.guard is not None:
                    visit(v, env.get(v, frozenset()) | {did})
                    env[v] = env.get(v, frozenset()) | {did}
                else:
                    env[v] = frozenset([did])
        for e in term_exprs(b.term):
            for x in e.walk():
                if isinstance(x, Var) and x.var in cand:
                    visit(x.var, env.get(x.var, frozenset()))
        return env

    order = rpo(F)
    IN = {a: {} for a in order}
    IN[F.entry] = {v: frozenset([d]) for v, d in entry.items() if v in cand}
    OUT = {}
    changed = True
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
        for a in order:
            b = F.blocks[a]
            if a != F.entry:
                env = {}
                for p in b.preds:
                    for v, ds in OUT.get(p, {}).items():
                        env[v] = env.get(v, frozenset()) | ds
                IN[a] = env
            out = walk_block(b, dict(IN[a]), lambda v, ds: None)
            if OUT.get(a) != out:
                OUT[a] = out
                changed = True

    def unite(v, ds):
        ds = list(ds)
        for d in ds[1:]:
            union(ds[0], d)
    for a in order:
        walk_block(F.blocks[a], dict(IN[a]), unite)
    # one variable per group
    groups = {}
    for v in cand:
        for d in defs_of[v]:
            groups.setdefault(v, {}).setdefault(find(d), []).append(d)
    newvar = {}                  # root did -> VarInfo
    split = False
    for v, gs in groups.items():
        if len(gs) == 1:
            continue
        keep = find(entry[v]) if v in entry else max(gs, key=lambda r: len(gs[r]))
        for r in gs:
            if r == keep:
                newvar[r] = v
                continue
            nv = copy.copy(v)
            nv.is_param = False
            nv.is_input = False
            nv.param_index = None
            nv.def_tys, nv.use_tys, nv.def_srcs = list(v.def_tys), list(v.use_tys), list(v.def_srcs)
            F.vars.append(nv)
            newvar[r] = nv
            split = True
        newvar[keep] = v
    if not split:
        return False

    def rename(env):
        def f(x):
            if isinstance(x, Var) and x.var in cand:
                ds = env.get(x.var)
                if ds:
                    nv = newvar.get(find(next(iter(ds))))
                    if nv is not None and nv is not x.var:
                        return Var(nv, x.ty)
            return x
        return lambda e: map_expr(e, f)
    for a in order:
        b = F.blocks[a]
        env = dict(IN[a])
        for st in b.stmts:
            fn = rename(env)
            if isinstance(st, Assign):
                st.src = fn(st.src)
                if st.guard is not None:
                    st.guard = fn(st.guard)
                if not isinstance(st.dst, Var):
                    st.dst = fn(st.dst)
            else:
                st.map(fn)
            v = stmt_defvar(st)
            if v in cand:
                did = site[id(st)]
                nv = newvar.get(find(did))
                if nv is not None and nv is not v:
                    st.dst = Var(nv, st.dst.ty)
                env[v] = (env.get(v, frozenset()) | {did}) if st.guard is not None else frozenset([did])
        if b.term is not None:
            b.term.map(rename(env))
    return True


def refresh_inputs(F):
    """After DCE a register first read by a (now removed) rotation copy is no
    longer read before being written: drop its input status and rename it."""
    def vs(e):
        return {x.var for x in e.walk() if isinstance(x, Var)}
    blocks = F.block_list()
    live_in = {b.addr: set() for b in blocks}
    changed = True
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
        for b in reversed(blocks):
            live = set()
            for sa in b.succs:
                live |= live_in.get(sa, set())
            for e in term_exprs(b.term):
                live |= vs(e)
            for st in reversed(b.stmts):
                v = stmt_defvar(st)
                if v is not None and st.guard is None:
                    live.discard(v)
                for e in st.exprs():
                    live |= vs(e)
                if isinstance(st, Assign) and not isinstance(st.dst, Var):
                    live |= vs(st.dst)
            if live != live_in[b.addr]:
                live_in[b.addr] = live
                changed = True
    entry = live_in.get(F.entry, set())
    defs, _ = collect(F)
    renamed = False
    for v in F.vars:
        if v.is_input and not v.is_param and v.reg != "sp" and v not in entry and v in defs:
            v.is_input = False
            renamed = True
    if renamed:
        name_vars(F)


def simplify_all(F):
    for b in F.block_list():
        for s in b.stmts:
            s.map(simplify)
        keep = []
        for s in b.stmts:
            g = s.guard
            if isinstance(g, Const):
                if g.value:
                    s.guard = None
                elif not any(has_call(e) for e in s.exprs()):
                    continue                      # never executed
            keep.append(s)
        b.stmts = keep
        if b.term is not None:
            b.term.map(simplify)
