"""Reverse if-conversion on the structured AST.

lcc turns branches into predicated straight-line code and schedules the
statements of both sides together; the structurer then prints every
predicated statement on its own:

    p41 = p37 && p39 && p33;
    if ( p41 ) v24 = local_20;
    if ( p37 && p39 && !p33 ) { error(...); return a1; }
    if ( p41 ) v14 = 31;
    if ( p66 ) v2 = 2;
    if ( !p66 ) { ... }

This pass rebuilds the branches:

* statements under the same condition are gathered into one `if` when the
  statements in between commute with them;
* a condition and its negation become if / else;
* conditions sharing conjuncts are nested (`if (a && b) X; if (a && !b) Y;`
  -> `if (a) { if (b) X; else Y; }`);
* conditions are simplified with what is known at that point: the enclosing
  conditions, the negation of the condition of an earlier early exit
  (`if (c) return;`), and the definitions of predicate variables
  (`p41 = p37 && p39 && p33` makes p41 <-> p37 && p39 && p33); a condition
  that cannot hold drops its statements, one that always holds drops the `if`.

Two items commute when they touch disjoint variables and do not both touch
memory (one of them writing), or when their conditions exclude each other
(at most one of them runs), whatever they do, provided neither changes what
the other's condition reads.  Nothing moves across a label.

Reasoning is propositional over "atoms": predicate variables and
comparisons, a comparison and its negation being one atom.  A small DPLL
decides the queries; the clause sets are kept to the atoms a query reaches.
"""
from .ir import Var, Un, Bin, Const, Assign, Store, Call, Load, Intr
from .simplify import simplify, repr_key, NEG

MAX_BACK = 80          # how far a statement is moved up to join a group
MAX_ATOMS = 96         # beyond that a query is answered "don't know"
MAX_STEPS = 4000       # DPLL search budget per query (then: "don't know")


def _S():
    m = _S.__dict__.get("m")
    if m is None:              # imported late: structure imports this module
        from . import structure as m
        _S.m = m
    return m


# ------------------------------------------------------------ conditions
POS = {"==", "<s", "<=s", "<u", "<=u"}


class Lit:
    __slots__ = ("key", "pol", "e", "vars", "mem")

    def __init__(self, e):
        pol = True
        while isinstance(e, Un) and e.op == "!":
            e, pol = e.a, not pol
        if isinstance(e, Bin) and e.op in NEG and e.op not in POS and NEG[e.op] in POS:
            e, pol = Bin(NEG[e.op], e.a, e.b, e.ty), not pol
        self.e, self.pol = e, pol
        self.key = repr_key(e)
        self.vars = frozenset(x.var for x in e.walk() if isinstance(x, Var))
        self.mem = any(isinstance(x, (Load, Call)) or (isinstance(x, Intr) and not x.pure) for x in e.walk())

    def neg(self):
        n = Lit.__new__(Lit)
        n.e, n.pol, n.key, n.vars, n.mem = self.e, not self.pol, self.key, self.vars, self.mem
        return n

    def expr(self):
        return self.e if self.pol else simplify(Un("!", self.e, "b"))

    def sig(self):
        return (self.key, self.pol)


def conj(cond):
    """cond -> list of Lit (a conjunction), duplicates removed"""
    out, seen = [], set()

    def walk(e):
        if isinstance(e, Bin) and e.op == "&&":
            walk(e.a)
            walk(e.b)
            return
        if isinstance(e, Un) and e.op == "!" and isinstance(e.a, Bin) and e.a.op == "||":
            walk(simplify(Un("!", e.a.a, "b")))
            walk(simplify(Un("!", e.a.b, "b")))
            return
        if isinstance(e, Const) and e.value:
            return
        lit = Lit(e)
        if lit.sig() not in seen:
            seen.add(lit.sig())
            out.append(lit)
    walk(cond)
    return out


def disj(cond):
    """cond -> list of Lit when cond is a disjunction of literals, else None"""
    if isinstance(cond, Bin) and cond.op == "||":
        a, b = disj(cond.a), disj(cond.b)
        return a + b if a is not None and b is not None else None
    if isinstance(cond, Bin) and cond.op == "&&":
        return None
    return [Lit(cond)]


def mk(lits):
    if not lits:
        return Const(1, "b")
    e = lits[0].expr()
    for l in lits[1:]:
        e = Bin("&&", e, l.expr(), "b")
    return e


def lit_vars(lits):
    out = set()
    for l in lits:
        out |= l.vars
    return out


# ------------------------------------------------------------ SAT
class Facts:
    """clauses (frozensets of (atom key, polarity)) known to hold here, with
    the variables each atom reads (to drop facts when one is redefined)"""

    def __init__(self, clauses=(), avars=None, amem=None):
        self.clauses = list(clauses)
        self.avars = dict(avars or {})
        self.amem = dict(amem or {})

    def copy(self):
        return Facts(self.clauses, self.avars, self.amem)

    def note(self, lits):
        for l in lits:
            self.avars[l.key] = l.vars
            self.amem[l.key] = l.mem

    def add(self, lits_with_pol):
        """add one clause: a disjunction of Lit"""
        self.note(lits_with_pol)
        c = frozenset(l.sig() for l in lits_with_pol)
        if c and c not in self.clauses:
            self.clauses.append(c)

    def kill(self, defs, store):
        """forget what mentions a redefined variable (or memory, after a store)"""
        if not defs and not store:
            return
        bad = {k for k, vs in self.avars.items() if (vs & defs) or (store and self.amem.get(k))}
        if bad:
            self.clauses = [c for c in self.clauses if not any(k in bad for k, _ in c)]

    def unsat(self, lits):
        """is facts & lits (a conjunction) unsatisfiable?  False when unsure"""
        if not lits:
            return False
        units = {}
        for l in lits:
            if units.get(l.key, l.pol) != l.pol:
                return True
            units[l.key] = l.pol
        # clauses reachable from the query atoms
        atoms = set(units)
        cl = []
        rest = list(self.clauses)
        grown = True
        while grown:
            grown = False
            keep = []
            for c in rest:
                if any(k in atoms for k, _ in c):
                    cl.append(c)
                    for k, _ in c:
                        if k not in atoms:
                            atoms.add(k)
                    grown = True
                else:
                    keep.append(c)
            rest = keep
            if len(atoms) > MAX_ATOMS:
                return False
        budget = [MAX_STEPS]
        r = _dpll(cl, units, budget)
        return r is True

    def implies(self, lit, others):
        return self.unsat(list(others) + [lit.neg()])


def _dpll(clauses, assign, budget):
    """True: unsatisfiable, False: satisfiable, None: out of budget"""
    budget[0] -= 1
    if budget[0] < 0:
        return None
    assign = dict(assign)
    clauses = list(clauses)
    while True:
        unit = None
        new = []
        for c in clauses:
            sat = False
            free = []
            for k, p in c:
                v = assign.get(k)
                if v is None:
                    free.append((k, p))
                elif v == p:
                    sat = True
                    break
            if sat:
                continue
            if not free:
                return True                 # a clause is false: unsatisfiable
            if len(free) == 1:
                unit = free[0]
            new.append(free)
        if not new:
            return False                    # all satisfied
        if unit is None:
            break
        assign[unit[0]] = unit[1]
        clauses = new
    k = new[0][0][0]
    for val in (True, False):
        a = dict(assign)
        a[k] = val
        r = _dpll(new, a, budget)
        if r is not True:
            return r
    return True


# ------------------------------------------------------------ node effects
class Info:
    __slots__ = ("defs", "uses", "load", "store", "jump", "other_jump", "label")

    def __init__(self):
        self.defs, self.uses = set(), set()
        self.load = self.store = self.jump = self.other_jump = self.label = False


def _expr_info(e, inf):
    for x in e.walk():
        if isinstance(x, Var):
            inf.uses.add(x.var)
        elif isinstance(x, Load):
            inf.load = True
        elif isinstance(x, Call):
            inf.load = inf.store = True
        elif isinstance(x, Intr) and not x.pure:
            inf.load = inf.store = True


def info(n, inf=None, loop_depth=0):
    S = _S()
    inf = inf or Info()
    if isinstance(n, S.SStmt):
        s = n.s
        for e in s.exprs():
            _expr_info(e, inf)
        if isinstance(s, Assign):
            if isinstance(s.dst, Var):
                inf.defs.add(s.dst.var)
            else:
                inf.store = True
                _expr_info(s.dst, inf)
        elif isinstance(s, Store):
            inf.store = True
    elif isinstance(n, S.SIf):
        _expr_info(n.cond, inf)
        for x in n.then + n.els:
            info(x, inf, loop_depth)
    elif isinstance(n, S.SLoop):
        if n.cond is not None:
            _expr_info(n.cond, inf)
        for s in (n.init, n.step):
            if s is not None:
                info(S.SStmt(s), inf, loop_depth)
        for x in n.body:
            info(x, inf, loop_depth + 1)
    elif isinstance(n, (S.SBreak, S.SContinue)):
        if loop_depth == 0:
            inf.jump = inf.other_jump = True
    elif isinstance(n, S.SReturn):
        inf.jump = True
        if n.value is not None:
            _expr_info(n.value, inf)
    elif isinstance(n, S.SGoto):
        inf.jump = inf.other_jump = True
    else:                                   # labels, switch, raw text: never moved
        inf.label = True
        inf.jump = inf.other_jump = True
    return inf


def _touches(inf, lits):
    """does code with effects inf change what a condition (lits) reads?"""
    return bool(inf.defs & lit_vars(lits)) or (inf.store and any(l.mem for l in lits))


def _independent(a, b):
    if a.defs & (b.uses | b.defs) or b.defs & a.uses:
        return False
    if (a.store and (b.load or b.store)) or (b.store and a.load):
        return False
    return True


# ------------------------------------------------------------ the pass
class Pass:
    def __init__(self):
        self.changed = False
        self._info = {}

    def inf(self, n):
        r = self._info.get(id(n))
        if r is None or r[0] is not n:
            r = (n, info(n))
            self._info[id(n)] = r
        return r[1]

    def forget(self, n):
        self._info.pop(id(n), None)
        self._info.pop(("b", id(n)), None)

    # -- items: an SIf, or a statement with an implicit condition "true"
    @staticmethod
    def cond_lits(n):
        S = _S()
        if isinstance(n, S.SIf):
            if not hasattr(n, "_lits"):
                n._lits = conj(n.cond)
            return n._lits
        return None

    def set_cond(self, n, lits):
        n._lits = lits
        n.cond = mk(lits)
        self.forget(n)

    def facts_along(self, nodes, start):
        """facts before each item of nodes (index -> Facts)"""
        S = _S()
        out = []
        f = start.copy()
        for n in nodes:
            out.append(f)
            f = f.copy()
            inf = self.inf(n)
            f.kill(inf.defs, inf.store)
            if isinstance(n, S.SStmt) and isinstance(n.s, Assign) and isinstance(n.s.dst, Var) and \
                    n.s.guard is None and n.s.dst.ty == "b":
                self.define(f, n.s.dst, n.s.src)
            elif isinstance(n, S.SIf):
                lits = self.cond_lits(n)
                if lits and S.ends_jump(n.then) and not (n.els and S.ends_jump(n.els)) and \
                        not _touches(self.inf(n), lits):
                    if not n.els:
                        f.add([l.neg() for l in lits])          # past `if (c) return;`: !c
                    elif len(lits) == 1:
                        f.add([lits[0].neg()])
                elif lits and n.els and S.ends_jump(n.els) and not S.ends_jump(n.then) and \
                        not _touches(self.inf(n), lits):
                    for l in lits:
                        f.add([l])
        out.append(f)
        return out

    @staticmethod
    def define(f, dst, src):
        """p = c1 && c2 (or c1 || c2): p <-> the definition"""
        p = Lit(dst)
        if p.key in {Lit(x).key for x in src.walk() if isinstance(x, Var) and x.var is dst.var}:
            return
        d = disj(src)
        if d is not None and len(d) > 1:
            f.add([p.neg()] + d)
            for l in d:
                f.add([p, l.neg()])
            return
        c = conj(src)
        if not c:
            return
        for l in c:
            f.add([p.neg(), l])
        f.add([p] + [l.neg() for l in c])

    # -- commuting
    def commute(self, y, z, fz):
        """can item y move before item z (z directly before it)?"""
        iy, iz = self.inf(y), self.inf(z)
        if iy.label or iz.label:
            return False
        ly, lz = self.cond_lits(y), self.cond_lits(z)
        if ly is not None and lz is not None and ly and lz:
            ty = self._body_info(y)
            tz = self._body_info(z)
            if not _touches(tz, ly) and not _touches(ty, lz) and fz.unsat(ly + lz):
                return True                       # at most one of them runs
        if iy.other_jump or iz.other_jump:
            return False
        if not _independent(iy, iz):
            return False
        if iy.jump and iz.jump:
            return False
        if iy.jump:
            # y may return before z now: z must have no effect anyone sees
            # after a return (it only sets local variables)
            return not iz.store
        if iz.jump:
            # y now runs even when z returns: it must not store, call or load
            # (a load could fault where the original never got)
            return not (iy.store or iy.load)
        return True

    def _body_info(self, n):
        r = self._info.get(("b", id(n)))
        if r is None or r[0] is not n or r[2] != (len(n.then), len(n.els)):
            inf = Info()
            for x in n.then + n.els:
                info(x, inf)
            r = (n, inf, (len(n.then), len(n.els)))
            self._info[("b", id(n))] = r
        return r[1]

    # -- one statement list
    def merge(self, nodes, start):
        """gather items of the same / opposite / overlapping condition"""
        S = _S()
        done = False
        j = 1
        facts = None
        while j < len(nodes):
            y = nodes[j]
            ly = self.cond_lits(y)
            if ly is None or not ly or self.inf(y).label:
                j += 1
                continue
            if facts is None:
                facts = self.facts_along(nodes, start)
            merged = False
            i = j - 1
            while i >= 0 and j - i <= MAX_BACK:
                x = nodes[i]
                lx = self.cond_lits(x)
                if lx and not self.inf(x).label and self.try_merge(x, lx, y, ly, facts[i]):
                    del nodes[j]
                    merged = True
                    break
                if not self.commute(y, x, facts[i]):
                    break
                i -= 1
            if merged:
                done = True
                facts = None
                continue                             # nodes[j] is the next item now
            j += 1
        return done

    def try_merge(self, x, lx, y, ly, fx):
        """y (moved up to just after x) folded into x; True if done"""
        S = _S()
        sx, sy = {l.sig() for l in lx}, {l.sig() for l in ly}
        bx = self._body_info(x)
        if sx == sy and not y.els:
            if _touches(bx, ly):
                return False
            if not S.ends_jump(x.then):            # (else y can never run: dropped)
                x.then.extend(y.then)              # if (c) A; if (c) B;  ->  if (c) { A; B; }
            self.forget(x)
            self.changed = True
            return True
        if len(lx) == 1 and len(ly) == 1 and lx[0].key == ly[0].key and lx[0].pol != ly[0].pol and \
                not y.els:
            if _touches(bx, ly):
                return False
            if not (x.els and S.ends_jump(x.els)):
                x.els.extend(y.then)               # if (c) A; if (!c) B;  ->  if (c) A else B
            self.forget(x)
            self.changed = True
            return True
        common = sx & sy
        if common and not x.els and not y.els and (sx != common or sy != common):
            pl = [l for l in lx if l.sig() in common]
            if _touches(bx, pl):
                return False
            rx = [l for l in lx if l.sig() not in common]
            ry = [l for l in ly if l.sig() not in common]
            inner = []
            for rest, n in ((rx, x), (ry, y)):
                if rest:
                    sub = S.SIf(mk(rest), n.then, [])
                    sub._lits = rest
                    sub._atom = getattr(n, "_atom", False)
                    inner.append(sub)
                else:
                    inner.extend(n.then)
            x.then, x.els = inner, []              # if (a && b) A; if (a && c) B;
            self.set_cond(x, pl)                   #   -> if (a) { if (b) A; if (c) B; }
            self.changed = True
            return True
        return False

    def simplify_seq(self, nodes, start):
        """conditions under what is known at their place"""
        S = _S()
        facts = self.facts_along(nodes, start)
        out = []
        done = False
        for k, n in enumerate(nodes):
            lits = self.cond_lits(n)
            if lits:
                f = facts[k]
                if not self.inf(n).label and f.unsat(lits):
                    out.extend(n.els)               # cannot hold here
                    done = True
                    continue
                keep = []
                for i, l in enumerate(lits):
                    others = keep + lits[i + 1:]
                    if f.implies(l, others):
                        continue
                    keep.append(l)
                if len(keep) != len(lits):
                    done = True
                    if not keep:
                        out.extend(n.then)          # always holds
                        continue
                    self.set_cond(n, keep)
            out.append(n)
        if done:
            nodes[:] = out
            self.changed = True
        return done

    def run(self, nodes, start):
        S = _S()
        atomize(nodes)
        for _ in range(4):
            a = self.merge(nodes, start)
            b = self.simplify_seq(nodes, start)
            if not (a or b):
                break
        facts = self.facts_along(nodes, start)
        for k, n in enumerate(nodes):
            if isinstance(n, S.SIf):
                lits = self.cond_lits(n)
                ft = facts[k].copy()
                for l in lits:
                    ft.add([l])
                self.run(n.then, ft)
                fe = facts[k].copy()
                if lits:
                    fe.add([l.neg() for l in lits])
                self.run(n.els, fe)
                if not n.then and n.els:
                    self.set_cond(n, [] if not lits else conj(simplify(Un("!", mk(lits), "b"))))
                    n.then, n.els = n.els, []
            elif isinstance(n, S.SLoop):
                inf = self.inf(n)
                fl = facts[k].copy()
                fl.kill(inf.defs, inf.store)
                self.run(n.body, fl)
        # empty ifs (their statements were all dropped)
        nodes[:] = [n for n in nodes if not (isinstance(n, S.SIf) and not n.then and not n.els
                                             and not self.inf(n).store and not self.inf(n).jump)]
        unatomize(nodes)


def atomize(nodes):
    """if ( g ) x = e;  ->  SIf(g, [x = e])"""
    S = _S()
    for k, n in enumerate(nodes):
        if isinstance(n, S.SStmt) and getattr(n.s, "guard", None) is not None:
            g = n.s.guard
            n.s.guard = None
            a = S.SIf(g, [n], [])
            a._atom = True
            nodes[k] = a


def unatomize(nodes):
    """an `if` left with one plain statement and no else prints as before"""
    S = _S()
    for k, n in enumerate(nodes):
        if isinstance(n, S.SIf) and len(n.then) == 1 and not n.els and isinstance(n.then[0], S.SStmt) \
                and getattr(n.then[0].s, "guard", None) is None and \
                not isinstance(n.then[0].s, (type(None),)):
            s = n.then[0].s
            if getattr(n, "_atom", False):
                s.guard = n.cond
                nodes[k] = n.then[0]
        if hasattr(n, "_lits"):
            del n._lits


def drop_dead_preds(nodes):
    """predicate variables nothing reads any more (their uses were folded)"""
    from .loops import count_vars
    S = _S()
    changed = True
    rounds = 0
    while changed and rounds < 20:
        rounds += 1
        changed = False
        uses = count_vars(nodes)["uses"]

        def walk(ns):
            nonlocal changed
            out = []
            for n in ns:
                if isinstance(n, S.SStmt) and isinstance(n.s, Assign) and isinstance(n.s.dst, Var) and \
                        n.s.dst.ty == "b" and not uses.get(n.s.dst.var) and not n.s.dst.var.is_param and \
                        not any(isinstance(x, (Call, Load)) or (isinstance(x, Intr) and not x.pure)
                                for x in n.s.src.walk()):
                    changed = True
                    continue
                if isinstance(n, S.SIf):
                    n.then = walk(n.then)
                    n.els = walk(n.els)
                elif isinstance(n, S.SLoop):
                    n.body = walk(n.body)
                elif isinstance(n, S.SSwitch):
                    n.cases = [(v, walk(b)) for v, b in n.cases]
                    if n.default is not None:
                        n.default = walk(n.default)
                out.append(n)
            return out
        nodes[:] = walk(nodes)


def _pure(e):
    return not any(isinstance(x, (Call, Load)) or (isinstance(x, Intr) and not x.pure) for x in e.walk())


def _replace_var(e, var, by):
    from .ir import map_expr
    return simplify(map_expr(e, lambda x: by if isinstance(x, Var) and x.var is var else x))


def inline_preds(nodes):
    """p = a && !b; ... if (p) X  ->  if (a && !b) X  when that is p's only use
    and nothing in between changes a or b"""
    from .loops import count_vars
    S = _S()
    cv = count_vars(nodes)
    uses, defs = cv["uses"], cv["defs"]
    done = False

    def subst_in(n, var, by):
        """put `by` for var where n reads it in a condition it owns; True if done"""
        if isinstance(n, S.SIf) and any(isinstance(x, Var) and x.var is var for x in n.cond.walk()):
            n.cond = _replace_var(n.cond, var, by)
            return True
        if isinstance(n, S.SStmt) and n.s.guard is not None and \
                any(isinstance(x, Var) and x.var is var for x in n.s.guard.walk()):
            n.s.guard = _replace_var(n.s.guard, var, by)
            return True
        return False

    def find(ns, start, var, src_vars, by):
        """the use of var in ns[start:], if it is reached before any change of
        src_vars: True (replaced), False (blocked / not found), None (go on)"""
        for k in range(start, len(ns)):
            n = ns[k]
            if subst_in(n, var, by):
                return True
            if isinstance(n, S.SIf):
                for sub in (n.then, n.els):
                    r = find(sub, 0, var, src_vars, by)
                    if r is not None:
                        return r
            inf = info(n)
            if var in inf.uses:
                return False                    # read some other way (loop, switch)
            if inf.defs & src_vars or inf.label:
                return False
        return None

    def walk(ns):
        nonlocal done
        k = 0
        while k < len(ns):
            n = ns[k]
            if isinstance(n, S.SStmt) and isinstance(n.s, Assign) and isinstance(n.s.dst, Var) and \
                    n.s.guard is None and n.s.dst.ty == "b" and defs.get(n.s.dst.var) == 1 and \
                    uses.get(n.s.dst.var) == 1 and _pure(n.s.src) and not n.s.dst.var.is_param:
                var = n.s.dst.var
                src_vars = {x.var for x in n.s.src.walk() if isinstance(x, Var)}
                if var not in src_vars and find(ns, k + 1, var, src_vars, n.s.src):
                    del ns[k]
                    done = True
                    continue
            if isinstance(n, S.SIf):
                walk(n.then)
                walk(n.els)
            elif isinstance(n, S.SLoop):
                walk(n.body)
            k += 1
    walk(nodes)
    return done


def fold_return(nodes):
    """v = e; return v;  ->  return e;"""
    S = _S()
    out = []
    for n in nodes:
        if isinstance(n, S.SIf):
            n.then = fold_return(n.then)
            n.els = fold_return(n.els)
        elif isinstance(n, S.SLoop):
            n.body = fold_return(n.body)
        if isinstance(n, S.SReturn) and isinstance(n.value, Var) and out and isinstance(out[-1], S.SIf) and \
                out[-1].els and not S.ends_jump(out[-1].then + out[-1].els[:0]) and _assigns_all(out[-1], n.value.var):
            # if (a) v = x; else v = y; return v;  ->  if (a) return x; else return y;
            _push_return(out[-1], n.value)
            continue
        if isinstance(n, S.SReturn) and isinstance(n.value, Var) and out and isinstance(out[-1], S.SStmt) and \
                isinstance(out[-1].s, Assign) and out[-1].s.guard is None and \
                isinstance(out[-1].s.dst, Var) and out[-1].s.dst.var is n.value.var and \
                not getattr(out[-1].s, "is_call", False):
            n = S.SReturn(out.pop().s.src)
        out.append(n)
    return out


def _assigns_all(n, var):
    """every path through if/else n ends with `var = ...` (and nothing jumps)"""
    S = _S()
    for br in (n.then, n.els):
        if not br:
            return False
        last = br[-1]
        if isinstance(last, S.SIf):
            if not last.els or not _assigns_all(last, var):
                return False
        elif not (isinstance(last, S.SStmt) and isinstance(last.s, Assign) and last.s.guard is None and
                  isinstance(last.s.dst, Var) and last.s.dst.var is var):
            return False
        if any(info(x).jump for x in br[:-1]):
            return False
    return True


def _push_return(n, v):
    S = _S()
    for br in (n.then, n.els):
        if isinstance(br[-1], S.SIf):
            _push_return(br[-1], v)
        else:
            br.append(S.SReturn(v))
            br[:] = fold_return(br)


def flatten_exits(nodes):
    """if (c) A else { ...; return; }  ->  if (!c) { ...; return; } A
    (and if (c) { ...; return; } else A  ->  if (c) { ...; return; } A):
    the error exits first, the main path unindented"""
    S = _S()
    out = []
    for n in nodes:
        if isinstance(n, S.SIf):
            n.then = flatten_exits(n.then)
            n.els = flatten_exits(n.els)
            if n.els and S.ends_jump(n.els) and not S.ends_jump(n.then) and \
                    not (len(n.els) == 1 and isinstance(n.els[0], S.SIf)):
                n.cond = simplify(Un("!", n.cond, "b"))
                n.then, n.els = n.els, n.then
            if n.els and S.ends_jump(n.then):
                rest = n.els
                n.els = []
                out.append(n)
                out.extend(rest)
                continue
        elif isinstance(n, S.SLoop):
            n.body = flatten_exits(n.body)
        out.append(n)
    return out


def _shape(ns):
    """structural key of a statement list (for identical exit blocks)"""
    S = _S()
    out = []
    for n in ns:
        if isinstance(n, S.SStmt):
            s = n.s
            out.append((type(s).__name__, repr_key(s.dst) if isinstance(s, Assign) else None,
                        tuple(repr_key(e) for e in s.exprs())))
        elif isinstance(n, S.SReturn):
            out.append(("ret", repr_key(n.value) if n.value is not None else None))
        elif isinstance(n, S.SIf):
            a, b = _shape(n.then), _shape(n.els)
            if a is None or b is None:
                return None
            out.append(("if", repr_key(n.cond), a, b))
        elif isinstance(n, (S.SBreak, S.SContinue)):
            out.append((type(n).__name__,))
        elif isinstance(n, S.SGoto):
            out.append(("goto", n.label))
        else:
            return None
    return tuple(out)


def merge_same_exits(nodes):
    """if (a) { E; return; } if (b) { E; return; }  ->  if (a || b) { E; return; }"""
    S = _S()
    out = []
    for n in nodes:
        if isinstance(n, S.SIf):
            n.then = merge_same_exits(n.then)
            n.els = merge_same_exits(n.els)
        elif isinstance(n, S.SLoop):
            n.body = merge_same_exits(n.body)
        if isinstance(n, S.SIf) and not n.els and out and isinstance(out[-1], S.SIf) and not out[-1].els and \
                S.ends_jump(n.then) and S.ends_jump(out[-1].then):
            a, b = _shape(out[-1].then), _shape(n.then)
            if a is not None and a == b:
                out[-1].cond = simplify(Bin("||", out[-1].cond, n.cond, "b"))
                continue
        out.append(n)
    return out


def unpredicate(body):
    p = Pass()
    p.run(body, Facts())
    for _ in range(3):
        drop_dead_preds(body)
        if not inline_preds(body):
            break
        p = Pass()
        p.run(body, Facts())
    drop_dead_preds(body)
    body[:] = merge_same_exits(flatten_exits(fold_return(body)))
    return body
