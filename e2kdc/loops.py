"""Loop recovery on the structured AST: for / while / do-while.

Runs after structure.cleanup().  All transforms are local rewrites of
statement lists and keep semantics; each one checks its preconditions and
does nothing when they fail.

  sink_temps     t = E(i); i += c; ... use(t) ...   ->  ... use(E(i)) ...; i += c;
  sink_into_cond do { p = E; S } while (p)          ->  do { S } while (E)
  unguard        [i = X;] if (C[i:=X]) do B while (C)  ->  while (C) B
                 (also when the loop is printed as while(X){..; if(C) continue; break;})
  make_for       i = X; while (C(i)) { B; i += c; }  ->  for (i = X; C; i += c) B

The guard test works on expressions: the loop condition with the values
assigned right before the guard substituted must simplify to the guard.
"""
from .ir import Assign, Var, Bin, Un, Const, Cast, Load, map_expr
from .simplify import simplify, repr_key


_S = []


def _structure():
    if not _S:                  # imported late: structure imports this module
        from . import structure
        _S.append(structure)
    return _S[0]


# ------------------------------------------------------------ expression walks
def node_exprs(n, deep=True):
    """Expressions *read* by node n (and, if deep, its children)."""
    S = _structure()
    if isinstance(n, S.SStmt):
        yield from n.s.exprs()
        d = getattr(n.s, "dst", None)
        if d is not None and not isinstance(d, Var):
            yield d
    elif isinstance(n, S.SIf):
        yield n.cond
        if deep:
            for x in n.then + n.els:
                yield from node_exprs(x)
    elif isinstance(n, S.SLoop):
        if n.cond is not None:
            yield n.cond
        for s in (getattr(n, "init", None), getattr(n, "step", None)):
            if s is not None:
                yield from s.exprs()
        if deep:
            for x in n.body:
                yield from node_exprs(x)
    elif isinstance(n, S.SSwitch):
        yield n.expr
        if deep:
            for _, b in n.cases:
                for x in b:
                    yield from node_exprs(x)
            for x in (n.default or []):
                yield from node_exprs(x)
    elif isinstance(n, S.SReturn):
        if n.value is not None:
            yield n.value
    elif isinstance(n, S.SRaw):
        if isinstance(n.text, tuple):
            yield n.text[1]


def node_defs(n):
    """Variables assigned anywhere inside node n."""
    S = _structure()
    out = set()
    if isinstance(n, S.SStmt):
        d = getattr(n.s, "dst", None)
        if isinstance(d, Var):
            out.add(d.var)
    elif isinstance(n, S.SIf):
        for x in n.then + n.els:
            out |= node_defs(x)
    elif isinstance(n, S.SLoop):
        for s in (getattr(n, "init", None), getattr(n, "step", None)):
            if s is not None and isinstance(getattr(s, "dst", None), Var):
                out.add(s.dst.var)
        for x in n.body:
            out |= node_defs(x)
    elif isinstance(n, S.SSwitch):
        for _, b in n.cases:
            for x in b:
                out |= node_defs(x)
        for x in (n.default or []):
            out |= node_defs(x)
    return out


def uses_in(n, var):
    c = 0
    for e in node_exprs(n):
        for x in e.walk():
            if isinstance(x, Var) and x.var is var:
                c += 1
    return c


def mentions(n, var):
    return uses_in(n, var) > 0 or var in node_defs(n)


def expr_vars(e):
    return {x.var for x in e.walk() if isinstance(x, Var)}


def subst_map(e, env):
    def f(x):
        if isinstance(x, Var) and x.var in env:
            return env[x.var]
        return x
    return map_expr(e, f)


def subst_node(n, var, rep):
    """Replace reads of var in node n (recursively) by rep."""
    S = _structure()
    fn = lambda e: simplify(subst_map(e, {var: rep}))
    if isinstance(n, S.SStmt):
        n.s.map(fn)
        d = getattr(n.s, "dst", None)
        if d is not None and not isinstance(d, Var):
            n.s.dst = fn(d)
    elif isinstance(n, S.SIf):
        n.cond = fn(n.cond)
        for x in n.then + n.els:
            subst_node(x, var, rep)
    elif isinstance(n, S.SLoop):
        if n.cond is not None:
            n.cond = fn(n.cond)
        for s in (getattr(n, "init", None), getattr(n, "step", None)):
            if s is not None:
                s.map(fn)
        for x in n.body:
            subst_node(x, var, rep)
    elif isinstance(n, S.SSwitch):
        n.expr = fn(n.expr)
        for _, b in n.cases:
            for x in b:
                subst_node(x, var, rep)
        for x in (n.default or []):
            subst_node(x, var, rep)
    elif isinstance(n, S.SReturn):
        if n.value is not None:
            n.value = fn(n.value)
    elif isinstance(n, S.SRaw) and isinstance(n.text, tuple):
        n.text = (n.text[0], fn(n.text[1]))


def plain_assign(n):
    """SStmt holding an unguarded `Var = expr`; returns the Assign or None."""
    S = _structure()
    if isinstance(n, S.SStmt) and isinstance(n.s, Assign) and n.s.guard is None \
            and isinstance(n.s.dst, Var) and not getattr(n.s, "is_call", False):
        return n.s
    return None


def is_step(s, var=None):
    """i = i (+|-) const  (the classic induction step)."""
    if s is None or not isinstance(s.dst, Var):
        return False
    v = s.dst.var
    if var is not None and v is not var:
        return False
    e = s.src
    return isinstance(e, Bin) and e.op in ("+", "-") and isinstance(e.a, Var) and e.a.var is v \
        and isinstance(e.b, Const)


def has_continue(nodes):
    """continue that targets the loop owning `nodes` (nested loops excluded)."""
    S = _structure()
    for n in nodes:
        if isinstance(n, S.SContinue):
            return True
        if isinstance(n, S.SIf) and (has_continue(n.then) or has_continue(n.els)):
            return True
        if isinstance(n, S.SSwitch):
            for _, b in n.cases:
                if has_continue(b):
                    return True
    return False


def has_labels(nodes):
    S = _structure()
    for n in nodes:
        if isinstance(n, S.SLabel):
            return True
        if isinstance(n, S.SIf) and (has_labels(n.then) or has_labels(n.els)):
            return True
        if isinstance(n, S.SLoop) and has_labels(n.body):
            return True
        if isinstance(n, S.SSwitch) and any(has_labels(b) for _, b in n.cases):
            return True
    return False


def contains_jump(n, in_loop=False):
    """Does executing node n possibly leave the straight line (break/continue of
    the current loop, return, goto, label)?"""
    S = _structure()
    if isinstance(n, (S.SReturn, S.SGoto, S.SLabel)):
        return True
    if isinstance(n, (S.SBreak, S.SContinue)):
        return not in_loop
    if isinstance(n, S.SRaw):
        return "unreachable" in str(n.text) or "goto" in str(n.text)
    if isinstance(n, S.SIf):
        return any(contains_jump(x, in_loop) for x in n.then + n.els)
    if isinstance(n, S.SLoop):
        return any(contains_jump(x, True) for x in n.body)
    if isinstance(n, S.SSwitch):
        return any(contains_jump(x, True) for _, b in n.cases for x in b)   # break = end of switch
    return False


def ekey(e):
    return repr_key(simplify(e))


def is_true(e):
    e = simplify(e)
    return isinstance(e, Const) and e.value != 0


# ------------------------------------------------------------ 1. temporaries
def node_writes_mem(n):
    """Does node n store to memory or call something?"""
    from .ir import Store, Call, Intr
    S = _structure()
    if isinstance(n, S.SStmt):
        if isinstance(n.s, Store) or getattr(n.s, "asm", None):
            return True
        return any(isinstance(x, Call) or (isinstance(x, Intr) and not x.pure)
                   for e in n.s.exprs() for x in e.walk())
    if isinstance(n, S.SIf):
        return any(node_writes_mem(x) for x in n.then + n.els)
    if isinstance(n, S.SLoop):
        return any(node_writes_mem(x) for x in n.body)
    if isinstance(n, S.SSwitch):
        return any(node_writes_mem(x) for _, b in n.cases for x in b)
    return isinstance(n, S.SRaw)


def pure(e):
    from .ir import Call, Intr
    return not any(isinstance(x, Call) or (isinstance(x, Intr) and not x.pure) for x in e.walk())


def cheap(e):
    """Small enough to duplicate: v, (T)v, v op c, ((T)v) op c."""
    if isinstance(e, Cast):
        e = e.a
    if isinstance(e, Var):
        return True
    if isinstance(e, Bin) and e.op in ("+", "-", "<<", "*") and isinstance(e.b, Const):
        a = e.a.a if isinstance(e.a, Cast) else e.a
        return isinstance(a, Var)
    return False


def sink_temps(nodes, ctx):
    """t = E; [i += c;] ...uses of t...   ->  ...uses of E...; [i += c;]
    E is pure; t is defined once and every use of t follows in this list.
    The step of a variable of E (typical: `v6 = v5; ++v5; a[v6]`) is moved
    after the last use, so the loop step ends up at the end of the body."""
    S = _structure()
    changed = True
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
        for k in range(len(nodes) - 1):
            a = plain_assign(nodes[k])
            if a is None:
                continue
            t, E = a.dst.var, a.src
            if t.is_input or t in expr_vars(E):
                continue
            if not pure(E):
                # v = call(); next-stmt(v)  ->  next-stmt(call()) : evaluation order kept
                # when the next statement uses v once and has no other side effects
                nx = nodes[k + 1]
                if ctx.defs.get(t, 0) == 1 and ctx.uses.get(t, 0) == 1 and isinstance(nx, S.SStmt) \
                        and uses_in(nx, t) == 1 and not node_writes_mem(nx) and nx.s.guard is None:
                    subst_node(nx, t, E)
                    del nodes[k]
                    ctx.refresh()
                    changed = True
                    break
                continue
            if ctx.defs.get(t, 0) != 1:
                continue
            total = ctx.uses.get(t, 0)
            if total == 0 or (total > 1 and not cheap(E)):
                continue
            if E.ty == "b" and sum(1 for _ in E.walk()) > 10:
                continue                # keep named predicates for big conditions
            ev = expr_vars(E)
            loads = E.has_load()
            step, si, last, seen = None, None, None, 0
            for m in range(k + 1, len(nodes)):
                n = nodes[m]
                st = plain_assign(n)
                if step is None and st is not None and is_step(st) and st.dst.var in ev \
                        and uses_in(n, t) == 0:
                    step, si = st.dst.var, m
                    continue
                d = node_defs(n)
                if t in d or (d & ev) or (step is not None and mentions(n, step)):
                    break
                if loads and node_writes_mem(n) and (uses_in(n, t) == 0 or not isinstance(n, S.SStmt)):
                    break               # a store/call may change what E reads
                u = uses_in(n, t)
                if u:
                    seen += u
                    if seen == total:
                        last = m
                        break
            if last is None:
                continue
            if si is not None and any(contains_jump(nodes[m]) for m in range(si + 1, last + 1)):
                continue                # the step would be skipped on that exit
            for m in range(k + 1, last + 1):
                subst_node(nodes[m], t, E)
            if si is not None:
                nodes.insert(last + 1, nodes[si])
                del nodes[si]
            del nodes[k]
            ctx.refresh()
            changed = True
            break
    return nodes


def sink_into_cond(loop, ctx):
    """do { ...; p = E; S } while (f(p))  ->  do { ...; S } while (f(E))
    when p is used only by the condition and S does not change E."""
    changed = True
    while changed and loop.cond is not None:
        changed = False
        cv = expr_vars(loop.cond)
        body = loop.body
        for q in range(len(body) - 1, -1, -1):
            a = plain_assign(body[q])
            if a is None:
                if any(v in cv for v in node_defs(body[q])):
                    break
                continue
            p, E = a.dst.var, a.src
            if p not in cv:
                continue
            if p.is_input or not pure(E) or ctx.defs.get(p, 0) != 1 or ctx.uses.get(p, 0) != 1:
                break
            ev = expr_vars(E)
            tail = body[q + 1:]
            # steps `i += c` of variables of E after p's definition are compensated:
            # p = f(i); i += c;  ->  cond uses f(i - c)
            shift = {}
            bad = False
            for n in tail:
                st = plain_assign(n)
                if st is not None and is_step(st) and st.dst.var in ev and st.dst.var not in shift \
                        and p not in node_defs(n):
                    shift[st.dst.var] = st
                    continue
                if node_defs(n) & (ev | {p}):
                    bad = True
                    break
            if bad:
                break
            if E.has_load() and any(node_writes_mem(n) for n in tail):
                break
            for v, st in shift.items():
                inv = Bin("-" if st.src.op == "+" else "+", Var(v), st.src.b, st.src.ty)
                E = subst_map(E, {v: inv})
            loop.cond = simplify(subst_map(loop.cond, {p: E}))
            del body[q]
            ctx.refresh()
            changed = True
            break



def sink_step(loop):
    """while (C(i)) { ...; i += c; S }  ->  while (C(i)) { ...; S; i += c; }
    when S does not touch i (lets make_for see the step)."""
    S = _structure()
    if loop.cond is None:
        return
    body = loop.body
    cv = expr_vars(loop.cond)
    for q in range(len(body) - 2, -1, -1):
        st = plain_assign(body[q])
        if st is not None and is_step(st) and st.dst.var in cv:
            i = st.dst.var
            if all(not mentions(n, i) and not contains_jump(n) for n in body[q + 1:]):
                body.append(body.pop(q))
            return


def flip_tail(body):
    """Loop body ending in  `if (c) { A; jump } J;`  (J = break/return/goto,
    A ends in a jump)  ->  `if (!c) J; A`  (a trailing continue of A dropped).
    Repeated while it applies, so nested tails unfold:
        if (a) { if (b) { x; continue; } return -1; } break;
     -> if (!a) break; if (!b) return -1; x;"""
    S = _structure()
    done = False
    while len(body) >= 2 and isinstance(body[-1], (S.SBreak, S.SReturn, S.SGoto)) \
            and isinstance(body[-2], S.SIf):
        n, j = body[-2], body[-1]
        if n.els or not n.then or not S.ends_jump(n.then) or has_labels(n.then):
            break
        a = list(n.then)
        if isinstance(a[-1], S.SContinue):
            if has_continue(a[:-1]):
                break
            a = a[:-1]
        if len(a) == 1 and isinstance(a[0], (S.SBreak, S.SReturn, S.SGoto, S.SContinue)):
            break        # if (c) return x; break;  flipped would flip back forever (print_all)
        body[-2:] = [S.SIf(simplify(Un("!", n.cond, "b")), [j], [])] + a
        done = True
    return done


# ------------------------------------------------------------ 2. loop guard
def as_dowhile(loop):
    """Express loop as (body, cond) of `do body while (cond)`, or None."""
    S = _structure()
    if loop.kind == "dowhile":
        return list(loop.body), loop.cond
    if loop.kind != "while" or loop.cond is None:
        return None
    body = loop.body
    head = S.SIf(simplify(Un("!", loop.cond, "b")), [S.SBreak()], [])
    if len(body) >= 2 and isinstance(body[-1], S.SBreak) and isinstance(body[-2], S.SIf) \
            and S.is_only(body[-2].then, S.SContinue) and not body[-2].els \
            and not has_continue(body[:-2]):
        return [head] + body[:-2], body[-2].cond
    if body and isinstance(body[-1], S.SIf) and not has_continue(body[:-1]):
        last = body[-1]
        if S.is_only(last.then, S.SContinue) and S.is_only(last.els, S.SBreak):
            return [head] + body[:-1], last.cond
        if S.is_only(last.then, S.SBreak) and not last.els:
            return [head] + body[:-1], simplify(Un("!", last.cond, "b"))
    return None


def entry_env(nodes, k):
    """Values the variables certainly hold right before nodes[k], from the
    straight-line assignments that precede it."""
    env = {}
    later_defs = set()
    for j in range(k - 1, -1, -1):
        a = plain_assign(nodes[j])
        if a is None:
            break
        v = a.dst.var
        if v not in env and v not in later_defs and not (expr_vars(a.src) & later_defs) \
                and not a.src.has_load():
            env[v] = a.src
        later_defs.add(v)
    return env


def guard_matches(guard, cond, env):
    if ekey(guard) == ekey(subst_map(cond, env)):
        return True
    return False


def unguard(nodes):
    S = _structure()
    out = []
    for k, n in enumerate(nodes):
        # if (G) { do B while (C) }  with G == C[env]
        if isinstance(n, S.SIf) and not n.els:
            real = [x for x in n.then if not isinstance(x, S.SLabel)]
            if len(real) == 1 and isinstance(real[0], S.SLoop) and len(n.then) == 1:
                dw = as_dowhile(real[0])
                if dw is not None:
                    body, cond = dw
                    env = entry_env(nodes, k)
                    if guard_matches(n.cond, cond, env):
                        out.append(S.SLoop("while", cond, body))
                        continue
        # do B while (C) whose first test is known true -> while (C) B
        if isinstance(n, S.SLoop) and n.kind == "dowhile" and not has_continue(n.body):
            env = entry_env(nodes, k)
            if env and is_true(subst_map(n.cond, env)):
                out.append(S.SLoop("while", n.cond, list(n.body)))
                continue
        out.append(n)
    return out


# ------------------------------------------------------------ 3. for
def make_for(nodes):
    S = _structure()
    k = 0
    while k < len(nodes):
        n = nodes[k]
        if not (isinstance(n, S.SLoop) and n.kind == "while" and n.cond is not None
                and n.body and getattr(n, "step", None) is None):
            k += 1
            continue
        st = plain_assign(n.body[-1])
        if st is None or not is_step(st) or has_continue(n.body[:-1]):
            k += 1
            continue
        i = st.dst.var
        if i not in expr_vars(n.cond):
            k += 1
            continue
        # init: the last assignment to i before the loop, nothing after it touches i
        init_at = None
        for j in range(k - 1, -1, -1):
            a = plain_assign(nodes[j])
            if a is not None and a.dst.var is i:
                init_at = j
                break
            if a is None or mentions(nodes[j], i):
                break
        if init_at is None:
            k += 1
            continue
        init = nodes[init_at].s
        moved = expr_vars(init.src)
        if any(node_defs(nodes[m]) & moved for m in range(init_at + 1, k)):
            k += 1
            continue
        n.kind = "for"
        n.init = init
        n.step = st
        n.body = n.body[:-1]
        del nodes[init_at]
        k = init_at + 1 if init_at < k else k + 1
    return nodes


# ------------------------------------------------------------ driver
class Ctx:
    def __init__(self, root):
        self.root = root
        self.refresh()

    def refresh(self):
        c = count_vars(self.root)
        self.uses, self.defs = c["uses"], c["defs"]


def count_vars(nodes):
    from .analysis import _var_reads
    uses, defs = {}, {}
    S = _structure()
    reads = []

    def walk(ns):
        for n in ns:
            for e in node_exprs(n, deep=False):
                _var_reads(e, reads)
            if reads:
                for v in reads:
                    uses[v] = uses.get(v, 0) + 1
                reads.clear()
            if isinstance(n, S.SStmt):
                d = getattr(n.s, "dst", None)
                if isinstance(d, Var):
                    defs[d.var] = defs.get(d.var, 0) + 1
                    if n.s.guard is not None:
                        defs[d.var] += 1
            elif isinstance(n, S.SIf):
                walk(n.then)
                walk(n.els)
            elif isinstance(n, S.SLoop):
                for s in (n.init, n.step):
                    if s is not None and isinstance(s.dst, Var):
                        defs[s.dst.var] = defs.get(s.dst.var, 0) + 1
                walk(n.body)
            elif isinstance(n, S.SSwitch):
                for _, b in n.cases:
                    walk(b)
                walk(n.default or [])
    walk(nodes)
    return {"uses": uses, "defs": defs}


def _assume(nodes, start, c, truth):
    """C has the value `truth` from nodes[start] on, until a variable of C is
    redefined: fold C (and !C) wherever it appears, nested bodies included."""
    S = _structure()
    c = simplify(c)
    kc, kn = repr_key(c), repr_key(simplify(Un("!", c, "b")))
    cv = expr_vars(c)
    if c.has_load() or isinstance(c, Const):
        return

    def fix(e):
        def f(x):
            if x.ty == "b":
                kx = repr_key(x)
                if kx == kc:
                    return Const(1 if truth else 0, "b")
                if kx == kn:
                    return Const(0 if truth else 1, "b")
            return x
        return simplify(map_expr(e, f))

    def fix_node(n):
        if isinstance(n, S.SIf):
            n.cond = fix(n.cond)
            for x in n.then + n.els:
                fix_node(x)
        elif isinstance(n, S.SLoop):
            if n.cond is not None:
                n.cond = fix(n.cond)
            for x in n.body:
                fix_node(x)
        elif isinstance(n, S.SStmt):
            n.s.map(fix)
        elif isinstance(n, S.SReturn) and n.value is not None:
            n.value = fix(n.value)
    for m in range(start, len(nodes)):
        nm = nodes[m]
        if node_defs(nm) & cv:
            break
        fix_node(nm)


def _drop_const_guards(nodes):
    out = []
    for n in nodes:
        S = _structure()
        if isinstance(n, S.SStmt) and isinstance(n.s.guard, Const):
            if n.s.guard.value:
                n.s.guard = None
            elif not node_writes_mem(n):
                continue
        out.append(n)
    nodes[:] = out


def coalesce_copies(nodes, ctx):
    """t = E; v = t;  ->  v = E   (t has no other definition or use)"""
    k = 0
    changed = False
    while k + 1 < len(nodes):
        a, b = plain_assign(nodes[k]), plain_assign(nodes[k + 1])
        if a is not None and b is not None and isinstance(b.src, Var) and b.src.var is a.dst.var \
                and b.src.ty == a.dst.ty and b.dst.ty == a.dst.ty and a.dst.var is not b.dst.var \
                and ctx.uses.get(a.dst.var) == 1 and ctx.defs.get(a.dst.var) == 1 \
                and not getattr(a.dst.var, "fixed_name", None):
            b.src = a.src
            del nodes[k]
            changed = True
            continue
        k += 1
    if changed:
        ctx.refresh()
    return nodes


def sink_into_exit(nodes):
    """if (C) x = E;  ...;  if (C) { ...; return/break/goto }
    ->  ...;  if (C) { x = E; ...; return/... }
    when nothing in between reads x or changes x, C or E (an if-converted
    assignment that only feeds the exit path)."""
    S = _structure()
    changed = True
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
        for k, n in enumerate(nodes):
            if not (isinstance(n, S.SStmt) and isinstance(n.s, Assign) and isinstance(n.s.dst, Var)
                    and n.s.guard is not None and pure(n.s.src) and not getattr(n.s, "is_call", False)):
                continue
            x = n.s.dst.var
            g = simplify(n.s.guard)
            kg = repr_key(g)
            if x in expr_vars(g) or x in expr_vars(n.s.src):
                continue
            fixed = expr_vars(g) | expr_vars(n.s.src) | {x}
            load = n.s.src.has_load()
            for m in range(k + 1, len(nodes)):
                nm = nodes[m]
                if isinstance(nm, S.SIf) and not nm.els and nm.then and S.ends_jump(nm.then) \
                        and repr_key(simplify(nm.cond)) == kg:
                    del nodes[k]
                    nm.then.insert(0, S.SStmt(Assign(Var(x, n.s.dst.ty), n.s.src, None)))
                    fold_returns(nm.then)
                    changed = True
                    break
                if uses_in(nm, x) or (node_defs(nm) & fixed) or _leaves(nm) or \
                        isinstance(nm, (S.SLabel, S.SReturn)) or (load and node_writes_mem(nm)):
                    break
            if changed:
                break
    return nodes


def assume_after_exit(nodes):
    """if (C) { ...; return/break/continue/goto }  ->  C is false in the rest of
    the list (nested bodies too), until a variable of C is redefined.
    if (C) {A} else {B}  ->  C is true inside A, false inside B."""
    S = _structure()
    for k, n in enumerate(nodes):
        if isinstance(n, S.SIf):
            _assume(n.then, 0, n.cond, True)
            _assume(n.els, 0, n.cond, False)
            _drop_const_guards(n.then)
            _drop_const_guards(n.els)
        if not (isinstance(n, S.SIf) and not n.els and n.then and S.ends_jump(n.then)):
            continue
        _assume(nodes, k + 1, n.cond, False)
    _drop_const_guards(nodes)
    return nodes


def fold_returns(nodes):
    """v = X; return v;  ->  return X;   (the function ends, v is never read again)"""
    S = _structure()
    k = 0
    while k + 1 < len(nodes):
        a = plain_assign(nodes[k])
        r = nodes[k + 1]
        if a is not None and isinstance(r, S.SReturn) and isinstance(r.value, Var) \
                and r.value.var is a.dst.var:
            src = a.src
            if r.value.ty != src.ty:
                from .analysis import convert
                src = convert(src, r.value.ty)
            nodes[k:k + 2] = [S.SReturn(simplify(src))]
            continue
        k += 1
    return nodes


def _rec(nodes, ctx):
    """Bottom-up over statement lists.  Lists are edited in place so that the
    tree ctx.root keeps pointing at what we change (for use counting)."""
    S = _structure()
    for n in nodes:
        if isinstance(n, S.SIf):
            _rec(n.then, ctx)
            _rec(n.els, ctx)
        elif isinstance(n, S.SLoop):
            _rec(n.body, ctx)
            if flip_tail(n.body):
                _rec(n.body, ctx)
            if n.kind == "dowhile":
                sink_into_cond(n, ctx)
        elif isinstance(n, S.SSwitch):
            for _, b in n.cases:
                _rec(b, ctx)
    sink_temps(nodes, ctx)
    new = unguard(nodes)
    if len(new) != len(nodes) or any(a is not b for a, b in zip(new, nodes)):
        nodes[:] = new
        ctx.refresh()
        for n in nodes:   # loops created by unguard: their bodies changed shape
            if isinstance(n, S.SLoop):
                sink_temps(n.body, ctx)
    for n in nodes:
        if isinstance(n, S.SLoop) and n.kind == "while":
            sink_step(n)
            sink_temps(n.body, ctx)
    make_for(nodes)
    coalesce_copies(nodes, ctx)
    sink_into_exit(nodes)
    assume_after_exit(nodes)
    fold_returns(nodes)
    ctx.refresh()
    return nodes


def _leaves(n, in_loop=False):
    """Can control leave node n sideways (goto/break/continue to a place that
    may read our variables)?  Returns are fine: their value is checked by uses."""
    S = _structure()
    if isinstance(n, S.SGoto):
        return True
    if isinstance(n, (S.SBreak, S.SContinue)):
        return not in_loop
    if isinstance(n, S.SRaw):
        return "goto" in str(n.text)
    if isinstance(n, S.SIf):
        return any(_leaves(x, in_loop) for x in n.then + n.els)
    if isinstance(n, S.SLoop):
        return any(_leaves(x, True) for x in n.body)
    if isinstance(n, S.SSwitch):
        return any(_leaves(x, True) for _, b in n.cases for x in b) or \
            any(_leaves(x, True) for x in (n.default or []))
    return False


def drop_overwritten(nodes):
    """x = E; ...(no read of x, no sideways exit)...; x = F  ->  drop the first
    assignment (E pure).  Typically a reload left over from a predicated path."""
    S = _structure()
    for n in nodes:
        if isinstance(n, S.SIf):
            drop_overwritten(n.then)
            drop_overwritten(n.els)
        elif isinstance(n, S.SLoop):
            drop_overwritten(n.body)
        elif isinstance(n, S.SSwitch):
            for _, b in n.cases:
                drop_overwritten(b)
    dead = set()
    for k, n in enumerate(nodes):
        a = plain_assign(n)
        if a is None or not pure(a.src):
            continue
        v = a.dst.var
        for m in range(k + 1, len(nodes)):
            nm = nodes[m]
            if uses_in(nm, v) or _leaves(nm) or isinstance(nm, S.SLabel):
                break
            b = plain_assign(nm)
            if b is not None and b.dst.var is v:
                dead.add(k)
                break
            if v in node_defs(nm):
                break
    if dead:
        nodes[:] = [n for k, n in enumerate(nodes) if k not in dead]
    return nodes


def recover_loops(nodes, F=None):
    nodes = list(nodes)
    ctx = Ctx(nodes)
    _rec(nodes, ctx)
    from .devec import devectorize, devectorize_maps
    if devectorize(nodes) | devectorize_maps(nodes, F):   # packed SIMD -> scalar lanes
        ctx.refresh()
        _rec(nodes, ctx)
    from .ivcanon import canon_counted
    names = {v.name for v in list(ctx.uses) + list(ctx.defs)}
    canon_counted(nodes, ctx, names)
    ctx.refresh()
    _each_list(nodes, lambda ns: coalesce_copies(ns, ctx))
    drop_overwritten(nodes)
    return nodes


def _each_list(nodes, fn):
    S = _structure()
    for n in nodes:
        if isinstance(n, S.SIf):
            _each_list(n.then, fn)
            _each_list(n.els, fn)
        elif isinstance(n, S.SLoop):
            _each_list(n.body, fn)
        elif isinstance(n, S.SSwitch):
            for _, b in n.cases:
                _each_list(b, fn)
    fn(nodes)
