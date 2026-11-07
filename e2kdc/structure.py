"""CFG simplification and structuring into if/while/do-while (goto fallback)."""
from .ir import Branch, Jump, Return, IJump, Unreachable, Bin, Un, Assign, ExprStmt, Store, Switch, map_expr, Const
from .simplify import simplify, repr_key
from .analysis import rpo, dominators
from .cfg import link


# ------------------------------------------------------------ AST
class N:
    pass


class SStmt(N):
    def __init__(self, s):
        self.s = s


class SIf(N):
    def __init__(self, cond, then, els=None):
        self.cond, self.then, self.els = cond, then, els or []


class SLoop(N):
    def __init__(self, kind, cond, body):
        self.kind, self.cond, self.body = kind, cond, body   # kind: while / dowhile / for
        self.init = self.step = None                          # for: Assign stmts


class SBreak(N):
    pass


class SContinue(N):
    pass


class SGoto(N):
    def __init__(self, label):
        self.label = label


class SLabel(N):
    def __init__(self, label):
        self.label = label


class SReturn(N):
    def __init__(self, value):
        self.value = value


class SRaw(N):
    def __init__(self, text):
        self.text = text


class SSwitch(N):
    def __init__(self, expr, cases, default):
        self.expr, self.cases, self.default = expr, cases, default


# ------------------------------------------------------------ CFG cleanups
def remove_empty_jumps(F):
    changed = True
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
        for b in F.block_list():
            if b.addr == F.entry or b.stmts or not isinstance(b.term, Jump):
                continue
            t = b.term.target
            if t == b.addr:
                continue
            for p in list(b.preds):
                pb = F.blocks[p]
                retarget(pb, b.addr, t)
            if b.preds:
                del F.blocks[b.addr]
                F.order.remove(b.addr)
                link(F)
                changed = True
                break
    prune_unreachable(F)


def retarget(pb, old, new):
    t = pb.term
    if isinstance(t, Jump) and t.target == old:
        t.target = new
    elif isinstance(t, Branch):
        if t.t == old:
            t.t = new
        if t.f == old:
            t.f = new
        if t.t == t.f:
            pb.term = Jump(t.t)
    elif isinstance(t, (Switch, IJump)) and getattr(t, "targets", None):
        t.targets = [new if x == old else x for x in t.targets]


def prune_unreachable(F):
    seen = set()
    stack = [F.entry]
    while stack:
        a = stack.pop()
        if a in seen or a not in F.blocks:
            continue
        seen.add(a)
        stack.extend(F.blocks[a].succs)
    for a in list(F.blocks):
        if a not in seen:
            del F.blocks[a]
            F.order.remove(a)
    link(F)


def fix_branches(F):
    from .ir import Const
    for b in F.block_list():
        t = b.term
        if isinstance(t, Branch) and isinstance(simplify(t.cond), Const):
            b.term = Jump(t.t if simplify(t.cond).value else t.f)     # e.g. ? %MLOCK
            continue
        if isinstance(t, Branch):
            if t.t not in F.blocks and t.f in F.blocks:
                b.term = Jump(t.f)
            elif t.f not in F.blocks and t.t in F.blocks:
                b.term = Jump(t.t)
            elif t.t == t.f:
                b.term = Jump(t.t)
        elif isinstance(t, Jump) and t.target not in F.blocks:
            b.term = Unreachable("jump outside function")
    # identical trivial return blocks on both sides of a branch
    for b in F.block_list():
        t = b.term
        if isinstance(t, Branch):
            x, y = F.blocks.get(t.t), F.blocks.get(t.f)
            if x and y and not x.stmts and not y.stmts and isinstance(x.term, Return) \
                    and isinstance(y.term, Return) and same_ret(x.term, y.term):
                b.term = Jump(t.t)
    link(F)


def same_ret(a, b):
    if a.value is None and b.value is None:
        return True
    if a.value is None or b.value is None:
        return False
    try:
        return a.value.key() == b.value.key()
    except Exception:
        return False


def merge_chains(F):
    changed = True
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
        for b in F.block_list():
            if isinstance(b.term, Jump):
                s = F.blocks.get(b.term.target)
                if s and s.addr != b.addr and len(s.preds) == 1 and s.addr != F.entry:
                    b.stmts.extend(s.stmts)
                    b.term = s.term
                    del F.blocks[s.addr]
                    F.order.remove(s.addr)
                    link(F)
                    changed = True
                    break


def combine_conditions(F):
    """Fold empty-condition blocks into && / ||."""
    changed = True
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
        for b1 in F.block_list():
            t1 = b1.term
            if not isinstance(t1, Branch):
                continue
            for side in ("t", "f"):
                a2 = getattr(t1, side)
                b2 = F.blocks.get(a2)
                if b2 is None or b2 is b1 or b2.stmts or len(b2.preds) != 1 or not isinstance(b2.term, Branch):
                    continue
                t2 = b2.term
                other = t1.f if side == "t" else t1.t
                c1, c2 = t1.cond, t2.cond
                if side == "t":
                    if t2.f == other:
                        nb = Branch(Bin("&&", c1, c2, "b"), t2.t, other)
                    elif t2.t == other:
                        nb = Branch(Bin("&&", c1, Un("!", c2, "b"), "b"), t2.f, other)
                    else:
                        continue
                else:
                    if t2.t == other:
                        nb = Branch(Bin("||", c1, c2, "b"), other, t2.f)
                    elif t2.f == other:
                        nb = Branch(Bin("||", c1, Un("!", c2, "b"), "b"), other, t2.t)
                    else:
                        continue
                nb.cond = simplify(nb.cond)
                b1.term = nb
                del F.blocks[b2.addr]
                F.order.remove(b2.addr)
                link(F)
                changed = True
                break
            if changed:
                break


def _cond_vars(e):
    from .ir import Var
    return {x.var for x in e.walk() if isinstance(x, Var)}


def fold_known_branches(F):
    """A block whose only predecessor already branched on the same condition
    knows its value:  P: if (c) X else B;  B: if (c) Y else Z  ->  B: goto Z.
    lcc leaves such pairs after `ct %ctpr1 ? ~%pred0; ct %ctpr3 ? %pred0`."""
    changed = True
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
        for b in F.block_list():
            t = b.term
            if not isinstance(t, Branch) or len(b.preds) != 1:
                continue
            p = F.blocks.get(b.preds[0])
            if p is None or p is b or not isinstance(p.term, Branch):
                continue
            pt = p.term
            if (pt.t == b.addr) == (pt.f == b.addr):
                continue
            known = pt.t == b.addr            # value of pt.cond inside b
            cv = _cond_vars(t.cond)
            if any(getattr(getattr(s, "dst", None), "var", None) in cv for s in b.stmts):
                continue
            kc, kn = repr_key(simplify(pt.cond)), repr_key(simplify(Un("!", pt.cond, "b")))
            k = repr_key(simplify(t.cond))
            if k == kc:
                val = known
            elif k == kn:
                val = not known
            else:
                continue
            b.term = Jump(t.t if val else t.f)
            link(F)
            changed = True
            break


def drop_known_guards(F):
    """B's only predecessor branched on c: inside B, `if (c) S` / `if (!c) S`
    (until a variable of c is redefined) is unconditional / dead."""
    changed = False
    for b in F.block_list():
        if len(b.preds) != 1:
            continue
        p = F.blocks.get(b.preds[0])
        if p is None or p is b or not isinstance(p.term, Branch):
            continue
        pt = p.term
        if (pt.t == b.addr) == (pt.f == b.addr):
            continue
        known = pt.t == b.addr
        c = simplify(pt.cond)
        kc, kn = repr_key(c), repr_key(simplify(Un("!", c, "b")))
        cv = _cond_vars(c)
        out = []
        stmts = b.stmts
        for i, st in enumerate(stmts):
            g = st.guard
            if g is not None:
                kg = repr_key(simplify(g))
                if kg in (kc, kn):
                    val = known if kg == kc else not known
                    changed = True
                    if val:
                        st.guard = None
                    elif not isinstance(st, Store) and not getattr(st, "asm", None) and \
                            not (isinstance(st, Assign) and getattr(st, "is_call", False)):
                        continue          # never executed here
                    else:
                        st.guard = Const(0, "b")
            out.append(st)
            if getattr(getattr(st, "dst", None), "var", None) in cv:
                out.extend(stmts[i + 1:])
                break
        b.stmts = out
    return changed


def sink_guarded(F):
    """Undo if-conversion at branch points:  B: ...; if (c) S; ...; branch c ? T : E
    with T reached only from B  ->  S (unguarded) at the top of T.  S moves only if
    nothing that stays after it in B reads what S writes, writes what S reads, or
    touches memory when S does."""
    from .ir import Var, Store, Call, Intr, Assign
    from .analysis import has_call

    def rd(st):
        out = set()
        for e in st.exprs():
            out |= _cond_vars(e)
        d = getattr(st, "dst", None)
        if d is not None and not isinstance(d, Var):
            out |= _cond_vars(d)
        return out

    def wr(st):
        d = getattr(st, "dst", None)
        return {d.var} if isinstance(d, Var) else set()

    def mem(st):
        return isinstance(st, Store) or any(has_call(e) or e.has_load() for e in st.exprs())

    moved_any = False
    for b in F.block_list():
        t = b.term
        if not isinstance(t, Branch) or t.t == t.f:
            continue
        kc, kn = repr_key(simplify(t.cond)), repr_key(simplify(Un("!", t.cond, "b")))
        sides = {}
        for side in (t.t, t.f):
            blk = F.blocks.get(side)
            if blk is not None and blk is not b and len(blk.preds) == 1:
                sides[side] = blk
        if not sides:
            continue
        later_r, later_w, later_mem = set(_cond_vars(t.cond)), set(), False
        keep, moved = [], {a: [] for a in sides}
        for st in reversed(b.stmts):
            g = st.guard
            tgt = None
            if g is not None and not getattr(st, "is_call", False):
                k = repr_key(simplify(g))
                tgt = t.t if k == kc else t.f if k == kn else None
            if tgt in sides and not (wr(st) & (later_r | later_w)) and not (rd(st) & later_w) \
                    and not (mem(st) and later_mem):
                st.guard = None
                moved[tgt].insert(0, st)
                continue
            keep.insert(0, st)
            later_r |= rd(st)
            later_w |= wr(st)
            later_mem = later_mem or mem(st)
        if any(moved.values()):
            b.stmts = keep
            for a, sts in moved.items():
                sides[a].stmts[0:0] = sts
            moved_any = True
    return moved_any


def _conj_keys(e):
    e = simplify(e)
    if isinstance(e, Bin) and e.op == "&&":
        return _conj_keys(e.a) | _conj_keys(e.b)
    return {repr_key(e)}


def sink_guarded_far(F):
    """`if (g) x = E` in B0  ->  `x = E` at the top of the first block T (dominated
    by B0) where the branches between B0 and T prove every conjunct of g, when
      - no path from B0 comes back to B0 before T,
      - every use of x is dominated by T (other paths never read this x),
      - between S and T nothing redefines x or the variables of g, E and the
        deciding conditions (and nothing writes memory if E reads it).
    This removes the predicated initialisations lcc hoists to the entry
    (`if (p) a5 = 0`), which otherwise look like extra parameters."""
    from .ir import Var, Store, Assign
    from .analysis import collect, dominators, dominates, has_call, stmt_defvar, rpo
    changed_any = False
    for _ in range(40):
        idom = dominators(F)
        defs, uses = collect(F)
        order = rpo(F)
        changed = False

        # predicates defined once by a plain assignment: expand  (!p14 -> !(p2 && !p3 && !p9))
        single = {}
        for v, ds in defs.items():
            if len(ds) == 1 and not v.is_input:
                db_, di_ = ds[0]
                st_ = db_.stmts[di_]
                if isinstance(st_, Assign) and st_.guard is None and v.ty == "b" and not st_.src.has_load():
                    single[v] = st_.src

        def expand(e, depth=0):
            if depth > 3:
                return e
            return simplify(map_expr(e, lambda x: expand(single[x.var], depth + 1)
                                     if isinstance(x, Var) and x.var in single else x))

        def stable(e):
            return all((len(defs.get(v, [])) <= 1 and not v.is_input) or (v.is_param and not defs.get(v))
                       for v in _cond_vars(e))

        def known(T, b0):
            """conjunct keys known true on entry to T: branches between B0 and T, and
            above B0 those whose variables never change."""
            from .ir import Const
            conds, cvars = [], set()
            cur, past = T, False
            while cur in idom and idom[cur] != cur and idom[cur] is not None:
                d = idom[cur]
                db, cb = F.blocks[d], F.blocks[cur]
                if isinstance(db.term, Branch) and len(cb.preds) == 1 and db.term.t != db.term.f:
                    c = db.term.cond if cur == db.term.t else \
                        simplify(Un("!", db.term.cond, "b")) if cur == db.term.f else None
                    if c is not None and (not past or stable(c)):
                        conds.append(simplify(c))
                        if not past:
                            cvars |= _cond_vars(c)
                if d == b0:
                    past = True
                cur = d
            conds = conds + [expand(c) for c in conds]
            atoms = set()
            for _ in range(3):
                for c in conds:
                    atoms |= _conj_keys(c)

                def fix(e):
                    if e.ty == "b" and not (isinstance(e, Bin) and e.op in ("&&", "||")):
                        if repr_key(e) in atoms:
                            return Const(1, "b")
                        if repr_key(simplify(Un("!", e, "b"))) in atoms:
                            return Const(0, "b")
                    return e
                conds = [simplify(map_expr(c, fix)) if isinstance(c, Bin) and c.op == "||" else c
                         for c in conds]
            for c in conds:
                atoms |= _conj_keys(c)
            return atoms, cvars

        for b0 in F.block_list():
            for si, st in enumerate(b0.stmts):
                if not (isinstance(st, Assign) and isinstance(st.dst, Var) and st.guard is not None) \
                        or getattr(st, "is_call", False) or has_call(st.src):
                    continue
                x = st.dst.var
                need_items = []

                def flat(e, depth=0):
                    e = simplify(e)
                    if isinstance(e, Bin) and e.op == "&&":
                        flat(e.a, depth)
                        flat(e.b, depth)
                    elif isinstance(e, Var) and e.var in single and depth < 3 and \
                            isinstance(simplify(single[e.var]), Bin) and simplify(single[e.var]).op == "&&":
                        flat(single[e.var], depth + 1)     # p16 = p2 && !p3 && p7
                    else:
                        need_items.append((repr_key(e), e))
                flat(st.guard)
                xuses = [(ub, ui) for ub, ui in uses.get(x, []) if not (ub is b0 and ui == si)]
                for ta in order:
                    if ta == b0.addr or not dominates(idom, b0.addr, ta):
                        continue
                    if not all(dominates(idom, ta, ub.addr) for ub, ui in xuses):
                        continue
                    have, cvars = known(ta, b0.addr)
                    if not all(k in have or repr_key(expand(c)) in have or _conj_keys(expand(c)) <= have
                               for k, c in need_items):
                        continue
                    T = F.blocks[ta]
                    fwd, stack = set(), list(b0.succs)
                    while stack:
                        a = stack.pop()
                        if a in fwd or a == ta:
                            continue
                        fwd.add(a)
                        stack.extend(F.blocks[a].succs)
                    if b0.addr in fwd:
                        break                            # a cycle through B0
                    back, stack = set(), [p for p in T.preds]
                    while stack:
                        a = stack.pop()
                        if a in back:
                            continue
                        back.add(a)
                        stack.extend(F.blocks[a].preds)
                    between = [F.blocks[a] for a in (fwd & back)]
                    watch = _cond_vars(st.guard) | _cond_vars(st.src) | {x} | cvars
                    loads = st.src.has_load()
                    ok = True
                    for s2 in b0.stmts[si + 1:] + [s2 for blk in between for s2 in blk.stmts]:
                        v = stmt_defvar(s2)
                        if v is not None and v in watch:
                            ok = False
                            break
                        if loads and (isinstance(s2, Store) or any(has_call(e) for e in s2.exprs())):
                            ok = False
                            break
                    if not ok:
                        break                            # later T only see more of the same
                    del b0.stmts[si]
                    st.guard = None
                    T.stmts.insert(0, st)
                    changed = changed_any = True
                    break
                if changed:
                    break
            if changed:
                break
        if not changed:
            break
    return changed_any


def simplify_cfg(F):
    fix_branches(F)
    fold_known_branches(F)
    prune_unreachable(F)
    remove_empty_jumps(F)
    merge_chains(F)
    combine_conditions(F)
    fix_branches(F)
    merge_chains(F)


# ------------------------------------------------------------ structuring
class LoopCtx:
    def __init__(self, header, body, follow):
        self.header, self.body, self.follow = header, body, follow


class Structurer:
    def __init__(self, F):
        self.F = F
        self.order = rpo(F)
        self.idx = {a: i for i, a in enumerate(self.order)}
        self.idom = dominators(F)
        self.ipdom = self._postdom()
        self.loops = self._loops()
        self.emitted = set()
        self.labels = set()
        self.loopstack = []
        self.switch_depth = 0

    def _postdom(self):
        F = self.F
        EXIT = -1
        succ = {a: list(F.blocks[a].succs) or [EXIT] for a in F.blocks}
        pred = {a: [] for a in F.blocks}
        pred[EXIT] = []
        for a, ss in succ.items():
            for s in ss:
                pred.setdefault(s, []).append(a)
        # reverse post-order on reversed graph from EXIT
        seen, out = set(), []

        def dfs(n):
            stack = [(n, iter(pred.get(n, [])))]
            seen.add(n)
            while stack:
                x, it = stack[-1]
                for p in it:
                    if p not in seen:
                        seen.add(p)
                        stack.append((p, iter(pred.get(p, []))))
                        break
                else:
                    stack.pop()
                    out.append(x)
        dfs(EXIT)
        order = list(reversed(out))
        idx = {a: i for i, a in enumerate(order)}
        ipd = {EXIT: EXIT}

        def inter(a, b):
            while a != b:
                while idx[a] > idx[b]:
                    a = ipd[a]
                while idx[b] > idx[a]:
                    b = ipd[b]
            return a
        changed = True
        _rounds = 0
        while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
            _rounds += 1
            changed = False
            for n in order:
                if n == EXIT:
                    continue
                ss = [s for s in succ[n] if s in ipd]
                if not ss:
                    continue
                new = ss[0]
                for s in ss[1:]:
                    new = inter(s, new)
                if ipd.get(n) != new:
                    ipd[n] = new
                    changed = True
        return {a: (v if v != EXIT else None) for a, v in ipd.items() if a != EXIT}

    def _loops(self):
        F = self.F
        loops = {}
        for b in F.block_list():
            for s in b.succs:
                if s in self.idx and b.addr in self.idx and self._dom(s, b.addr):
                    body = loops.setdefault(s, {s})
                    stack = [b.addr]
                    while stack:
                        x = stack.pop()
                        if x in body:
                            continue
                        body.add(x)
                        stack.extend(F.blocks[x].preds)
        return loops

    def _dom(self, a, b):
        while True:
            if a == b:
                return True
            nb = self.idom.get(b)
            if nb is None or nb == b:
                return False
            b = nb

    def _follow(self, header, body):
        exits = []
        for x in sorted(body, key=lambda a: self.idx.get(a, 0)):
            for s in self.F.blocks[x].succs:
                if s not in body and s not in exits:
                    exits.append(s)
        if not exits:
            return None
        pd = self.ipdom.get(header)
        if pd in exits:
            return pd
        # prefer exit that is not a bare return when several
        for e in exits:
            if not self._terminal(e):
                return e
        return exits[0]

    def _terminal(self, a):
        b = self.F.blocks[a]
        return not b.succs

    def run(self):
        body = self.region(self.F.entry, None, frozenset())
        body = strip_labels(body, self.labels)
        body = cleanup(body)
        from .loops import recover_loops
        body = recover_loops(body, self.F)
        from .predicate import unpredicate
        body = unpredicate(body)
        from .reroll import reroll
        body = reroll(body)
        body = final_tidy(body)
        body = drop_trailing_return(body)
        return body

    def _small_return(self, a):
        b = self.F.blocks[a]
        return isinstance(b.term, (Return, Unreachable)) and len(b.stmts) <= 1 and not \
            any(isinstance(s, Assign) and getattr(s, "is_call", False) for s in b.stmts)

    def region(self, cur, follow, stops):
        out = []
        while cur is not None:
            loop = self.loopstack[-1] if self.loopstack else None
            if loop and cur == loop.follow:
                if self.switch_depth > loop.switch_depth:
                    self.labels.add(cur)
                    out.append(SGoto(cur))
                else:
                    out.append(SBreak())
                return out
            if loop and cur == loop.header and loop.started:
                out.append(SContinue())
                return out
            if cur == follow:
                return out
            if any(cur == l.follow or cur == l.header for l in self.loopstack[:-1]):
                self.labels.add(cur)
                out.append(SGoto(cur))
                return out
            if cur in stops:
                self.labels.add(cur)
                out.append(SGoto(cur))
                return out
            if cur in self.emitted:
                if self._small_return(cur):
                    out.extend(self.block_body(self.F.blocks[cur], label=False))
                    t = self.F.blocks[cur].term
                    out.append(SReturn(t.value) if isinstance(t, Return) else SRaw("__builtin_unreachable();"))
                    return out
                self.labels.add(cur)
                out.append(SGoto(cur))
                return out
            if cur in self.loops and (loop is None or loop.header != cur):
                body = self.loops[cur]
                fl = self._follow(cur, body)
                ctx = LoopCtx(cur, body, fl)
                ctx.started = False
                ctx.switch_depth = self.switch_depth
                self.loopstack.append(ctx)
                lbody = self.region(cur, None, frozenset())
                self.loopstack.pop()
                out.append(SLoop("while", None, lbody))
                if fl is None:
                    return out
                cur = fl
                continue
            if loop and cur == loop.header:
                loop.started = True
            self.emitted.add(cur)
            b = self.F.blocks[cur]
            out.extend(self.block_body(b))
            t = b.term
            if isinstance(t, Jump):
                cur = t.target
                continue
            if isinstance(t, Return):
                out.append(SReturn(t.value))
                return out
            if isinstance(t, Unreachable):
                out.append(SRaw("__builtin_unreachable(); /* %s */" % t.why))
                return out
            if isinstance(t, IJump):
                out.append(SRaw(("goto *(void *)(%s);", t.target)))
                return out
            if isinstance(t, Switch):
                m = self.ipdom.get(cur)
                if loop and m is not None and m not in loop.body:
                    m = None
                if m is not None and m in self.emitted:
                    m = None
                inner_stops = stops | ({follow} if follow is not None else set())
                groups = {}
                for k, tg in enumerate(t.targets):
                    groups.setdefault(tg, []).append(k)
                cases = []
                self.switch_depth += 1
                for tg in sorted(groups, key=lambda a: groups[a][0]):
                    if tg == m:
                        body = [SBreak()]
                    else:
                        body = self.region(tg, m, inner_stops)
                        if m is not None and not ends_jump(body):
                            body.append(SBreak())
                    cases.append((groups[tg], body))
                self.switch_depth -= 1
                out.append(SSwitch(t.index, cases, None))
                if m is None:
                    return out
                cur = m
                continue
            if isinstance(t, Branch):
                m = self.ipdom.get(cur)
                if loop and m is not None and m not in loop.body:
                    m = None
                if m is not None and (m in self.emitted):
                    m = None
                inner_stops = stops | ({follow} if follow is not None else set())
                if m is not None and m == t.t:
                    then = []
                    els = self.region(t.f, m, inner_stops)
                    out.append(SIf(simplify(Un("!", t.cond, "b")), els, []))
                elif m is not None and m == t.f:
                    then = self.region(t.t, m, inner_stops)
                    out.append(SIf(t.cond, then, []))
                else:
                    then = self.region(t.t, m, inner_stops)
                    els = self.region(t.f, m, inner_stops)
                    out.append(SIf(t.cond, then, els))
                if m is None:
                    return out
                cur = m
                continue
            return out
        return out

    def block_body(self, b, label=True):
        out = [SLabel(b.addr)] if label else []
        for s in b.stmts:
            out.append(SStmt(s))
        return out


# ------------------------------------------------------------ AST cleanups
def ends_jump(nodes):
    if not nodes:
        return False
    n = nodes[-1]
    if isinstance(n, (SBreak, SContinue, SGoto, SReturn)):
        return True
    if isinstance(n, SRaw) and "unreachable" in str(n.text):
        return True
    if isinstance(n, SIf) and n.els:
        return ends_jump(n.then) and ends_jump(n.els)
    return False


def cleanup(nodes, in_loop=False):
    out = []
    for n in nodes:
        if isinstance(n, SIf):
            n.then = cleanup(n.then, in_loop)
            n.els = cleanup(n.els, in_loop)
            if not n.then and n.els:
                n.cond = simplify(Un("!", n.cond, "b"))
                n.then, n.els = n.els, []
            if not n.then and not n.els:
                continue
            if n.els and ends_jump(n.then):
                els = n.els
                n.els = []
                out.append(n)
                out.extend(els)
                continue
            if n.els and ends_jump(n.els) and not ends_jump(n.then):
                # if (c) {A} else {B; return}  ->  if (!c) {B; return} A
                n.cond = simplify(Un("!", n.cond, "b"))
                then = n.then
                n.then, n.els = n.els, []
                out.append(n)
                out.extend(then)
                continue
            out.append(n)
        elif isinstance(n, SLoop):
            n.body = cleanup(n.body, True)
            out.append(loopify(n))
        elif isinstance(n, SSwitch):
            n.cases = [(ks, cleanup(b, in_loop)) for ks, b in n.cases]
            out.append(n)
        else:
            out.append(n)
    # drop code after unconditional jump
    res = []
    for n in out:
        res.append(n)
        if isinstance(n, (SBreak, SContinue, SGoto, SReturn)):
            # keep labels that follow (targets of gotos)
            pass
    return trim_after_jump(res)


def trim_after_jump(nodes):
    out = []
    dead = False
    for n in nodes:
        if dead and not isinstance(n, SLabel):
            continue
        dead = False
        out.append(n)
        if isinstance(n, (SBreak, SContinue, SGoto, SReturn)):
            dead = True
    return out


def is_only(nodes, cls):
    real = [n for n in nodes if not isinstance(n, SLabel)]
    return len(real) == 1 and isinstance(real[0], cls)


def loopify(loop):
    body = loop.body
    # trailing continue is implicit
    while body and isinstance(body[-1], SContinue):
        body = body[:-1]
    # while (1) { if (c) break; ... }
    real = [i for i, n in enumerate(body) if not isinstance(n, SLabel)]
    if real:
        i0 = real[0]
        n = body[i0]
        if isinstance(n, SIf) and is_only(n.then, SBreak) and not n.els and \
                not any(isinstance(x, SLabel) for x in body[:i0]):
            rest = body[:i0] + body[i0 + 1:]
            return SLoop("while", simplify(Un("!", n.cond, "b")), cleanup_continue(rest))
    # do { ... if (c) continue; break; } / do { ... if (c) break; } while(1)
    if body:
        last = body[-1]
        if isinstance(last, SIf):
            if is_only(last.then, SContinue) and is_only(last.els, SBreak):
                return SLoop("dowhile", last.cond, cleanup_continue(body[:-1]))
            if is_only(last.then, SBreak) and not last.els:
                return SLoop("dowhile", simplify(Un("!", last.cond, "b")), cleanup_continue(body[:-1]))
            if is_only(last.then, SBreak) and is_only(last.els, SContinue):
                return SLoop("dowhile", simplify(Un("!", last.cond, "b")), cleanup_continue(body[:-1]))
            if is_only(last.then, SContinue) and not last.els:
                pass
        if len(body) >= 2 and isinstance(body[-1], SBreak) and isinstance(body[-2], SIf) and \
                is_only(body[-2].then, SContinue) and not body[-2].els:
            return SLoop("dowhile", body[-2].cond, cleanup_continue(body[:-2]))
    loop.body = body
    return loop


def cleanup_continue(body):
    while body and isinstance(body[-1], SContinue):
        body = body[:-1]
    return body


def drop_trailing_return(nodes):
    if nodes and isinstance(nodes[-1], SReturn) and nodes[-1].value is None:
        return nodes[:-1]
    return nodes


def fold_const_ifs(nodes):
    """if (1) {A} else {B} -> A;  if (0) ... -> B  (conditions that became
    constants during loop recovery), then drop code after jumps."""
    out = []
    for n in nodes:
        if isinstance(n, SIf):
            n.then = fold_const_ifs(n.then)
            n.els = fold_const_ifs(n.els)
            if isinstance(n.cond, Const):
                out.extend(n.then if n.cond.value else n.els)
                continue
        elif isinstance(n, SLoop):
            n.body = fold_const_ifs(n.body)
        elif isinstance(n, SSwitch):
            n.cases = [(ks, fold_const_ifs(b)) for ks, b in n.cases]
        out.append(n)
    return trim_after_jump(out)


def used_labels(nodes, acc=None):
    acc = set() if acc is None else acc
    for n in nodes:
        if isinstance(n, SGoto):
            acc.add(n.label)
        elif isinstance(n, SIf):
            used_labels(n.then, acc)
            used_labels(n.els, acc)
        elif isinstance(n, SLoop):
            used_labels(n.body, acc)
        elif isinstance(n, SSwitch):
            for _, b in n.cases:
                used_labels(b, acc)
            used_labels(n.default or [], acc)
    return acc


def final_tidy(body):
    from .loops import drop_overwritten
    body = fold_const_ifs(body)
    drop_overwritten(body)
    return strip_labels(body, used_labels(body))


def strip_labels(nodes, used):
    out = []
    for n in nodes:
        if isinstance(n, SLabel):
            if n.label in used:
                out.append(n)
            continue
        if isinstance(n, SIf):
            n.then = strip_labels(n.then, used)
            n.els = strip_labels(n.els, used)
        elif isinstance(n, SLoop):
            n.body = strip_labels(n.body, used)
        elif isinstance(n, SSwitch):
            n.cases = [(ks, strip_labels(b, used)) for ks, b in n.cases]
        out.append(n)
    return out


def structure(F):
    simplify_cfg(F)
    return Structurer(F).run()
