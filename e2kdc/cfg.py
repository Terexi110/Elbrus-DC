"""Control-flow recovery: follows ctpr registers through the function and
builds basic blocks of IR statements."""
from .ir import (Assign, Branch, Jump, Return, IJump, Unreachable, Reg, Call, Sym, Switch,
                 Intr, Un, Const, ExprStmt, Comment)
from .lift import Lifter


class Block:
    def __init__(self, addr):
        self.addr = addr
        self.stmts = []
        self.term = None
        self.succs = []
        self.preds = []
        self.bundles = []

    @property
    def name(self):
        return "loc_%x" % self.addr

    def __repr__(self):
        return "<Block %x>" % self.addr


class CallInfo:
    def __init__(self, wbs, bn, addr):
        self.wbs, self.bn, self.addr = wbs, bn, addr


class FuncIR:
    def __init__(self, func):
        self.func = func
        self.name = func.name
        self.blocks = {}
        self.entry = None
        self.order = []
        self.warnings = []
        self.loop_bundles = 0
        self.frame = 0
        self.calls = []

    def block_list(self):
        return [self.blocks[a] for a in self.order]


def build(func, resolver, jtables=None):
    jtables = jtables or {}
    lifter = Lifter(func, resolver)
    bundles = {b.addr: b for b in func.bundles}
    addrs = [b.addr for b in func.bundles]
    nxt = {addrs[i]: (addrs[i + 1] if i + 1 < len(addrs) else None) for i in range(len(addrs))}
    F = FuncIR(func)
    lifted = {}
    in_state = {}
    succ_edges = {}
    # state: (rotation bases, %ctpr descriptors, known predicate values)
    work = [(func.addr, ((0, 0, 0, 0), (), ()))]
    if func.bundles and func.bundles[0].addr != func.addr:
        work = [(func.bundles[0].addr, ((0, 0, 0, 0), (), ()))]

    def merge(s1, s2):
        if s1 == s2:
            return s1
        bn = s1[0] if s1[0] == s2[0] else s1[0]
        facts = tuple(sorted(set(s1[2]) & set(s2[2])))
        d1, d2 = dict(s1[1]), dict(s2[1])
        # a %ctpr set on one path only: the other path never prepared it, so a
        # transfer through it can only come from the path that did (code reached
        # through an infeasible fall-through, `ct A ? p; ct B ? ~p`, keeps it)
        # (conflicting values become ("multi", candidates): the lattice is finite, merging stays monotone)
        d = {}
        for k in set(d1) | set(d2):
            if k not in d2:
                d[k] = d1[k]
            elif k not in d1:
                d[k] = d2[k]
            elif d1[k] == d2[k]:
                d[k] = d1[k]
            else:
                # different values on different paths: keep the candidates (a call
                # through it can still be named if only one of them is a function)
                c = list(d1[k][1] if d1[k][0] == "multi" else [d1[k]])
                for x in (d2[k][1] if d2[k][0] == "multi" else [d2[k]]):
                    if x not in c:
                        c.append(x)
                d[k] = ("multi", tuple(sorted(c, key=repr)))
        d = tuple(sorted(d.items(), key=lambda kv: kv[0]))
        return (bn, d, facts)

    def pred_test(cond):
        """ct ... ? %predN / ~%predN  ->  (name, value that takes the branch)"""
        if isinstance(cond, Reg) and cond.name.startswith("pred"):
            return cond.name, True
        if isinstance(cond, Un) and cond.op == "!" and isinstance(cond.a, Reg) and cond.a.name.startswith("pred"):
            return cond.a.name, False
        return None

    iters = 0
    while work and iters < 100000:
        iters += 1
        addr, st = work.pop()
        if addr not in bundles:
            continue
        old = in_state.get(addr)
        if old is not None:
            new = merge(old, st)
            if new == old:
                continue
            st = new
        in_state[addr] = st
        bn, ctprs, facts = st
        ctpr = dict(ctprs)
        known = dict(facts)
        b = bundles[addr]
        L = lifter.lift(b, bn)
        lifted[addr] = L
        writes = {x.dst.name for x in L.stmts if isinstance(x, Assign) and isinstance(x.dst, Reg)
                  and x.dst.name.startswith("pred")}
        if L.call or L.loop or L.setbp is not None or L.abp is not None:
            out_known = {}
        else:
            out_known = {k: v for k, v in known.items() if k not in writes}
        out_bn = tuple(L.setbn) + bn[3:] if L.setbn else bn          # (rbs, rsz, rcur, psz)
        if L.setbp is not None:
            out_bn = out_bn[:3] + (L.setbp,)
        succs = []
        ctrl = None
        if L.call:
            # %ctpr registers survive a call (saved in the chain stack)
            tgt = ctpr.get(L.call[0], ("unknown",))
            ctrl_call = (tgt, L.call[1], L.call[2])
        else:
            ctrl_call = None
        new_ctpr = dict(ctpr)
        new_ctpr.update(L.ctpr_set)
        taken_known = fall_known = out_known
        if L.ct:
            desc = dict(ctprs).get(L.ct[0], ("unknown",))
            cond = L.ct[1]
            pt = pred_test(cond) if cond is not None else None
            if pt is not None and pt[0] in known:
                # `ct A ? p; ...; ct B ? ~p` with p unchanged: the second one is
                # always taken, its fall-through is dead (and must not merge the
                # %ctpr values of that path into the code after it)
                cond = None if known[pt[0]] == pt[1] else False
            elif pt is not None and pt[0] not in writes:
                taken_known = dict(out_known, **{pt[0]: pt[1]})
                fall_known = dict(out_known, **{pt[0]: not pt[1]})
            if cond is False:
                ctrl = None
                if nxt.get(addr) is not None:
                    succs.append(nxt[addr])
            else:
                ctrl = ("ct", desc, cond)
                if desc[0] == "disp" and desc[1][0] == "label":
                    succs.append(desc[1][1])
                if desc[0] == "reg" and addr in jtables:
                    succs.extend(t for t in jtables[addr] if t not in succs)
                if cond is not None and nxt.get(addr) is not None:
                    succs.append(nxt[addr])
        elif L.ibranch:
            ctrl = ("ib", L.ibranch[0], L.ibranch[1])
            succs.append(L.ibranch[0])
            if L.ibranch[1] is not None and nxt.get(addr) is not None:
                succs.append(nxt[addr])
        else:
            if nxt.get(addr) is not None:
                succs.append(nxt[addr])
        lifted[addr] = (L, ctrl, ctrl_call, succs, ctprs)
        cp = tuple(sorted(new_ctpr.items()))
        for s in succs:
            k = fall_known if (s == nxt.get(addr) and ctrl is not None and ctrl[0] == "ct") else \
                (taken_known if ctrl is not None and ctrl[0] == "ct" else out_known)
            work.append((s, (out_bn, cp, tuple(sorted(k.items())))))
    if not lifted:
        return F
    # ---------------------------------------------------------- leaders
    visited = sorted(lifted)
    leaders = {visited[0]}
    for a in visited:
        L, ctrl, cc, succs, _ = lifted[a]
        if ctrl is not None:
            for s in succs:
                leaders.add(s)
            if nxt.get(a) in lifted:
                leaders.add(nxt[a])
        else:
            for s in succs:
                if s != nxt.get(a):
                    leaders.add(s)
    # predecessors count to split at merge points
    preds = {}
    for a in visited:
        for s in lifted[a][3]:
            preds.setdefault(s, set()).add(a)
    for a, ps in preds.items():
        if len(ps) > 1 or any(p >= a for p in ps):
            leaders.add(a)
    entry = func.addr if func.addr in lifted else visited[0]
    leaders.add(entry)
    # ---------------------------------------------------------- blocks
    cur = None
    for a in visited:
        if a in leaders or cur is None:
            cur = Block(a)
            F.blocks[a] = cur
            F.order.append(a)
        L, ctrl, cc, succs, ctprs = lifted[a]
        cur.bundles.append(a)
        if L.loop:
            F.loop_bundles += 1
        for g in L.getsp:
            if g < 0 and F.frame == 0:
                F.frame = -g
        cur.stmts.extend(L.stmts)
        for u in L.unknown:
            F.warnings.append("unknown op at %x: %s" % (a, u))
        if cc is not None:
            tgt, wbs, ccond = cc
            target = target_expr(tgt, a, F)
            ci = CallInfo(wbs, in_state[a][0], a)
            call = Call(target, [], "i64")
            call.info = ci
            st = Assign(Reg("r%d" % (2 * wbs), "i64"), call, ccond, addr=a)
            st.is_call = True
            cur.stmts.append(st)
            F.calls.append((cur, st))
        if ctrl is not None:
            kind = ctrl[0]
            if kind == "ct":
                desc, cond = ctrl[1], ctrl[2]
                fall = nxt.get(a) if nxt.get(a) in lifted else None
                if desc[0] == "ret":
                    term_t = ("ret",)
                elif desc[0] == "disp" and desc[1][0] == "label":
                    term_t = ("goto", desc[1][1])
                elif desc[0] == "disp":
                    term_t = ("tail", desc[1])
                elif desc[0] == "reg":
                    term_t = ("ijump", desc[1])
                elif desc[0] == "syscall":
                    term_t = ("syscall", desc[1])
                else:
                    term_t = ("unknown", desc)
                cur.term = (term_t, cond, fall)
            else:
                tgt, cond = ctrl[1], ctrl[2]
                fall = nxt.get(a) if nxt.get(a) in lifted else None
                cur.term = (("goto", tgt), cond, fall)
            cur = None
        else:
            n = nxt.get(a)
            if n is None or n not in lifted:
                cur.term = (("fall_off",), None, None)
                cur = None
            elif n in leaders:
                cur.term = (("goto", n), None, None)
                cur = None
    # ---------------------------------------------------------- terminators
    for blk in F.block_list():
        if blk.term is None:
            blk.term = (("fall_off",), None, None)
        t, cond, fall = blk.term
        stmts_after = []
        if t[0] == "ret":
            T = Return(Reg("r0", "any"))
        elif t[0] == "goto":
            T = Jump(t[1])
        elif t[0] == "tail":
            call = Call(target_expr(("disp", t[1]), blk.addr, F), [], "i64")
            call.info = CallInfo(0, None, blk.addr)
            call.tail = True
            st = Assign(Reg("r0", "i64"), call, None, addr=blk.addr)
            st.is_call = True
            stmts_after.append(st)
            F.calls.append((blk, st))
            T = Return(Reg("r0", "any"))
        elif t[0] == "ijump":
            T = IJump(t[1])
            T.at = blk.bundles[-1] if blk.bundles else blk.addr
            T.targets = jtables.get(T.at)
        elif t[0] == "syscall":
            stmts_after.append(ExprStmt(Intr("syscall", [Reg("r0", "i64"), Reg("r1", "i64"), Reg("r2", "i64"),
                                                     Reg("r3", "i64")], "i64", pure=False)))
            T = Return(Reg("r0", "any"))
        elif t[0] == "fall_off":
            T = Unreachable("falls off the end")
        else:
            T = IJump(Intr("unknown_target", [], "i64", pure=False))
        if cond is not None:
            # conditional transfer: build a tiny block for the taken side if needed
            if isinstance(T, Jump):
                blk.term = Branch(cond, T.target, fall)
            else:
                side = Block(blk.addr + 1)       # synthetic address (bundles are 8-aligned)
                side.stmts = stmts_after
                # a conditional tail call lives in the side block now
                moved = {id(x) for x in stmts_after}
                F.calls = [(side if id(c) in moved else b, c) for b, c in F.calls]
                side.term = T
                F.blocks[side.addr] = side
                F.order.append(side.addr)
                blk.term = Branch(cond, side.addr, fall)
            if fall is None:
                if isinstance(blk.term, Branch):
                    # falls off the function: treat the taken side as unconditional
                    blk.term = Jump(blk.term.t)
        else:
            blk.stmts.extend(stmts_after)
            blk.term = T
    F.entry = entry
    F.order.sort()
    link(F)
    return F


def call_desc(desc):
    """the descriptor a call goes through: of several candidates (paths that
    prepared %ctpr differently, the call guarded so only one of them calls),
    the only one that is a function"""
    if desc[0] != "multi":
        return desc
    fs = [x for x in desc[1] if x[0] == "disp" and x[1][0] == "func"] + \
        [x for x in desc[1] if x[0] == "reg"]
    return fs[0] if len(fs) == 1 else ("unknown",)


def target_expr(desc, addr, F):
    desc = call_desc(desc)
    if desc[0] == "disp":
        d = desc[1]
        if d[0] == "func":
            return Sym(d[1], 0, "func")
        if d[0] == "label":
            if getattr(F, "func", None) is not None and d[1] == F.func.addr:
                return Sym(F.func.name, 0, "func")      # recursive call (executable, no reloc)
            return Sym("loc_%x" % d[1], 0, "label")
        if d[0] == "abs":
            return Const(d[1], "i64")
        return Intr("call_target", [], "i64", pure=False)
    if desc[0] == "reg":
        return desc[1]
    return Intr("call_target", [], "i64", pure=False)


def link(F):
    for b in F.blocks.values():
        b.succs, b.preds = [], []
    for b in F.block_list():
        t = b.term
        if isinstance(t, Jump):
            b.succs = [t.target]
        elif isinstance(t, Branch):
            b.succs = [t.t, t.f] if t.t != t.f else [t.t]
        elif isinstance(t, Switch):
            b.succs = []
            for x in t.targets:
                if x not in b.succs:
                    b.succs.append(x)
        elif isinstance(t, IJump) and getattr(t, "targets", None):
            b.succs = []
            for x in t.targets:
                if x not in b.succs:
                    b.succs.append(x)
        else:
            b.succs = []
        b.succs = [s for s in b.succs if s in F.blocks]
        for s in b.succs:
            F.blocks[s].preds.append(b.addr)
