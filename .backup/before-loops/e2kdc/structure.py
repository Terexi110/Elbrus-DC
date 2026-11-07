"""CFG simplification and structuring into if/while/do-while (goto fallback)."""
from .ir import Branch, Jump, Return, IJump, Unreachable, Bin, Un, Assign, ExprStmt, Store, Switch
from .simplify import simplify
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
    while changed:
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
    for b in F.block_list():
        t = b.term
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
    while changed:
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
    while changed:
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


def simplify_cfg(F):
    fix_branches(F)
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
        while changed:
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
