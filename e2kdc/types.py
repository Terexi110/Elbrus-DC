"""Type recovery on top of the register-width types of analysis.assign_types.

* pointees(ast):  element type of pointer variables (`int *p`) from the loads
  and stores made through them, propagated along copies, `p + c` and pointer
  comparisons.  The printer then writes `p[i]` / `*p` and scales pointer
  arithmetic (emit.Printer.ptr_add).
* scale_down(e, s): byte offset e as a count of s-byte elements, or None.
"""
from math import gcd
from .ir import Var, Bin, Un, Const, Cast, Load, Store, Assign, sext, BITS, is_float


def _S():
    m = _S.__dict__.get("m")
    if m is None:              # imported late: structure imports this module
        from . import structure as m
        _S.m = m
    return m


def scale_down(e, s):
    """e / s when e is structurally a multiple of s (no rounding, no overflow change)."""
    if s == 1:
        return e
    if isinstance(e, Const):
        v = sext(e.value, BITS.get(e.ty, 64))
        return Const(v // s, e.ty) if v % s == 0 else None
    if isinstance(e, Bin) and e.op in ("*", "<<") and isinstance(e.b, Const) and \
            (e.op == "*" or e.b.value < 63):
        k = sext(e.b.value, BITS.get(e.ty, 64)) if e.op == "*" else 1 << e.b.value
        if k % s == 0:
            return e.a if k == s else Bin("*", e.a, Const(k // s, e.ty), e.ty)
        # (v*2) << 2 over 8: the rest of the factor from the operand
        g = gcd(abs(k), s) if k else s
        inner = scale_down(e.a, s // g)
        if inner is None:
            return None
        return inner if k // g == 1 else Bin("*", inner, Const(k // g, e.ty), e.ty)
    if isinstance(e, Bin) and e.op in ("+", "-"):
        a, b = scale_down(e.a, s), scale_down(e.b, s)
        if a is not None and b is not None:
            return Bin(e.op, a, b, e.ty)
        return None
    if isinstance(e, Un) and e.op == "-":
        a = scale_down(e.a, s)
        return Un("-", a, e.ty) if a is not None else None
    return None


def split_addr(addr):
    """addr -> (pointer VarInfo, byte offset expr) when addr = p, p + off, off + p."""
    if isinstance(addr, Var) and getattr(addr.var, "pointer", False):
        return addr.var, Const(0, "i64")
    if isinstance(addr, Bin) and addr.op == "+":
        for a, b in ((addr.a, addr.b), (addr.b, addr.a)):
            if isinstance(a, Var) and getattr(a.var, "pointer", False) and \
                    not (isinstance(b, Var) and getattr(b.var, "pointer", False)):
                return a.var, b
            if isinstance(a, Bin) and a.op == "+":
                r = split_addr(a)
                if r is not None:
                    return r[0], Bin("+", r[1], b, "i64")
    if isinstance(addr, Bin) and addr.op == "-" and isinstance(addr.a, Var) and \
            getattr(addr.a.var, "pointer", False):
        return addr.a.var, Un("-", addr.b, "i64")
    return None


CTYPE = {"i8": "char", "u8": "unsigned char", "i16": "short", "u16": "unsigned short",
         "i32": "int", "u32": "unsigned int", "i64": "long", "u64": "unsigned long",
         "f32": "float", "f64": "double"}
CSIZE = {"char": 1, "unsigned char": 1, "short": 2, "unsigned short": 2, "int": 4,
         "unsigned int": 4, "long": 8, "unsigned long": 8, "float": 4, "double": 8}


def _elem(ty):
    return ty if ty in ("i8", "i16", "i32", "i64", "f32", "f64") else None


def ir_ty(pointee):
    """element type of the IR (signedness dropped): u32 -> i32; pointers: i64"""
    if isinstance(pointee, tuple):
        return "i64"
    return "i" + pointee[1:] if pointee and pointee[0] == "u" else pointee


def var_ctype(w, depth=0, seen=()):
    """C type of a pointer variable (for the element type of a pointer to it)"""
    from .structs import rep
    S = rep(getattr(w, "struct", None))
    if S is not None and S.name:
        return "struct %s *" % S.name
    if id(w) in seen:
        return "void *"              # p = *p: a pointer to itself, no finite C type
    if getattr(w, "pointer", False) and depth < 3:
        el = pointee_ctype(w, depth + 1, seen + (id(w),))
        return el + ("*" if el.endswith("*") else " *")
    return "void *" if getattr(w, "pointer", False) else "long"


def pointee_ctype(v, depth=0, seen=()):
    pe = getattr(v, "pointee", None)
    pel = getattr(v, "ptr_elem", None)
    if pel is not None and (pe is None or ir_ty(pe) == "i64"):
        pe = ("ptr", pel)
    if isinstance(pe, tuple):
        el = pe[1]
        if isinstance(el, str):
            c = CTYPE.get(el, "char")
            return c + " *"
        return var_ctype(el, depth, seen + (id(v),))
    return CTYPE.get(pe or "", "char")


def elem_csize(ptr_ctype):
    """sizeof(*p) for a pointer C type string"""
    el = ptr_ctype[:-1].strip()
    if el.endswith("*"):
        return 8
    if el.startswith("struct "):
        from .structs import SIZES
        return SIZES.get(el[7:].strip()) or 0
    return CSIZE.get(el, 1)


def address_bases(nodes):
    """64-bit variables whose every use in the AST is the base of a load or store
    address (`p + 4 * i`) are pointers.  The CFG-level analysis can miss them when
    the loop that walked the array through another register was rewritten with
    an index (or the only other use, a versioning check, was dropped)."""
    from .devec import _roots, _exprs_of
    from .ir import Store
    base, other = {}, set()

    def terms(e, out):
        if isinstance(e, Bin) and e.op == "+" and e.ty == "i64":
            terms(e.a, out)
            terms(e.b, out)
        else:
            out.append(e)

    def addr(a):
        ts = []
        terms(a, ts)
        vs = [t for t in ts if isinstance(t, Var)]
        if len(vs) == 1 and vs[0].ty == "i64":
            base[vs[0].var] = base.get(vs[0].var, 0) + 1
            for t in ts:
                if t is not vs[0]:
                    walk(t)
        else:
            walk(a)

    def walk(e):
        if isinstance(e, Load):
            addr(e.addr)
            return
        if isinstance(e, Var):
            other.add(e.var)
            return
        for c in e.children():
            walk(c)
    for h in _roots(nodes):
        if isinstance(h, Store):
            addr(h.addr)
            walk(h.val)
            if h.guard is not None:
                walk(h.guard)
            continue
        if isinstance(h, Assign) and isinstance(h.dst, Var):
            other.add(h.dst.var)          # assigned: its value comes from elsewhere too
        for e in _exprs_of(h):
            if isinstance(e, str):
                continue
            walk(e)
    for v in base:
        if v not in other and v.ty == "i64" and not getattr(v, "pointer", False) and \
                not getattr(v, "funcptr", None) and not getattr(v, "is_float", False):
            v.pointer = True


def pointees(nodes, extra=()):
    """Set VarInfo.pointee for pointer variables used in the AST `nodes`."""
    from .devec import _roots, _exprs_of
    address_bases(nodes)
    votes = {}               # var -> {elem type: count}, None = inconsistent access
    links = []               # (var, var): same pointee expected

    def access(addr, ty, uns=None):
        r = split_addr(addr)
        if r is None:
            return
        v, off = r
        el = _elem(ty)
        if el is None or scale_down(off, BITS[el] // 8) is None:
            votes.setdefault(v, {})["?"] = votes.get(v, {}).get("?", 0) + 1
            return
        d = votes.setdefault(v, {})
        d[el] = d.get(el, 0) + 1
        if uns is not None and not is_float(el):
            k = ("u" if uns else "s") + el
            d[k] = d.get(k, 0) + 1

    def ptr_of(e):
        if isinstance(e, Var) and getattr(e.var, "pointer", False):
            return e.var
        if isinstance(e, Bin) and e.op in ("+", "-"):
            return ptr_of(e.a) if not ptr_of(e.b) or e.op == "-" else ptr_of(e.b)
        return None

    def deref_base(addr):
        """the pointer value of an address `L + off` whose base is a loaded pointer L"""
        terms = []

        def t(x):
            if isinstance(x, Bin) and x.op == "+":
                t(x.a)
                t(x.b)
            else:
                terms.append(x)
        t(addr)
        loads = [x for x in terms if isinstance(x, Load) and x.ty == "i64"]
        return loads[0] if len(loads) == 1 and not any(isinstance(x, Var) and getattr(x.var, "pointer", False)
                                                        for x in terms) else None

    def elem_ptr(load, code):
        r = split_addr(load.addr)
        if r is not None and getattr(r[0], "ptr_elem", None) is None:
            r[0].ptr_elem = code     # v[i][j]: v holds pointers to `code`

    def walk(e, parent=None):
        if isinstance(e, Load) or (isinstance(e, Cast) and isinstance(e.a, Load)):
            ld = e if isinstance(e, Load) else e.a
            b = deref_base(ld.addr)
            if b is not None and _elem(ld.ty):
                el = ld.ty
                if isinstance(e, Cast) and not e.signed and not is_float(el) and \
                        not (isinstance(parent, Bin) and parent.op in ("==", "!=")):
                    el = "u" + el[1:]
                elem_ptr(b, el)
        if isinstance(e, Cast) and isinstance(e.a, Load) and BITS.get(e.ty, 0) > BITS.get(e.a.ty, 64):
            # extension tells the signedness, unless the value is only tested
            # for (in)equality or truncated again (strlen, string copy)
            agnostic = (isinstance(parent, Bin) and parent.op in ("==", "!=")) or \
                (isinstance(parent, Cast) and BITS.get(parent.ty, 64) <= BITS[e.a.ty]) or parent == "store8"
            access(e.a.addr, e.a.ty, None if agnostic else not e.signed)
            for c in e.a.children():
                walk(c, e.a)
            return
        if isinstance(e, Load):
            access(e.addr, e.ty)
        if isinstance(e, Bin) and e.op in ("==", "!=", "<u", "<=u", ">u", ">=u", "<s", "<=s", ">s", ">=s", "-"):
            a, b = ptr_of(e.a), ptr_of(e.b)
            if a is not None and b is not None and a is not b:
                links.append((a, b))
            elif e.op != "-" and (a is None) != (b is None):
                # p != end: the other side is a pointer too
                o = e.b if a is not None else e.a
                if isinstance(o, Var) and o.ty == "i64" and o.var.ty == "i64" and \
                        not getattr(o.var, "pointer", False) and not getattr(o.var, "funcptr", None):
                    o.var.pointer = True
                    links.append((a or b, o.var))
        for c in e.children():
            walk(c, e)

    roots = _roots(nodes) + list(extra)
    for h in roots:
        if isinstance(h, Store):
            access(h.addr, h.ty)
            b = deref_base(h.addr)
            if b is not None and _elem(h.ty):
                elem_ptr(b, h.ty)
        if isinstance(h, Assign) and isinstance(h.dst, Var) and getattr(h.dst.var, "pointer", False):
            p = ptr_of(h.src)
            if p is not None and p is not h.dst.var:
                links.append((h.dst.var, p))
        for e in _exprs_of(h):
            walk(e, "store8" if isinstance(h, Store) and e is h.val and BITS.get(h.ty, 64) <= 8 else None)
    ptype = {}
    for v, d in votes.items():
        kinds = sorted((k for k in d if k != "?" and k[0] not in "su"), key=lambda k: -d[k])
        if kinds and (len(kinds) == 1 or d[kinds[0]] > d[kinds[1]]):
            el = kinds[0]                # majority; other accesses keep explicit casts
            if d.get("u" + el) and not d.get("s" + el):
                el = "u" + el[1:]        # only ever zero-extended: unsigned elements
            ptype[v] = el
        else:
            ptype[v] = None              # mixed: bytes (char *)
    # propagate along copies / comparisons to pointers without accesses
    changed = True
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
        for a, b in links:
            for x, y in ((a, b), (b, a)):
                if x not in ptype and ptype.get(y) is not None:
                    ptype[x] = ptype[y]
                    changed = True
    for v, t in ptype.items():
        v.pointee = t
    return ptype


def apply_hints(F, ptype):
    """pointer variables without accesses: the library prototype's pointee"""
    for v in F.vars:
        if getattr(v, "pointer", False) and v not in ptype and getattr(v, "pointee_hint", None):
            v.pointee = v.pointee_hint


def elem_size(v):
    t = getattr(v, "pointee", None)
    if getattr(v, "ptr_elem", None) is not None:
        return 8
    return BITS[ir_ty(t)] // 8 if t else 1


# ------------------------------------------------------------ int vs long
#
# lcc computes `int` arithmetic in 64-bit registers wherever signed overflow
# is undefined (loop counters, sums), so after lifting such variables are
# `long`.  A 64-bit variable is turned back into a 32-bit one when every value
# it receives is a 32-bit value: a sign-extended int ("s"), a zero-extended
# unsigned ("u"), a small constant (fits both), or +,-,* / selects of
# sign-extended values (no signed overflow, as in the source).  Variables
# that mix "s" and "u" values stay 64-bit.  A parameter becomes `int` when the
# code truncates it somewhere ((int)a1) and gives it no 64-bit value.

RET32 = {}          # function name -> returns a 32-bit value (pass-1 summaries)


def _kind(e, K):
    """frozenset of {'s','u'} (empty: fits both) for a 32-bit valued e, or None."""
    from .ir import Select, Un, Intr, FConst, Call, Sym
    if isinstance(e, Call) and isinstance(e.target, Sym) and RET32.get(e.target.name):
        return frozenset("s")
    if e.ty == "b":
        return frozenset()
    if e.ty in ("i32", "i16", "i8"):
        return frozenset("s")
    if isinstance(e, Const):
        v = sext(e.value, BITS.get(e.ty, 64))
        if 0 <= v < 2 ** 31:
            return frozenset()
        if -2 ** 31 <= v < 0:
            return frozenset("s")
        if 0 <= v < 2 ** 32:
            return frozenset("u")
        return None
    if isinstance(e, Var):
        if e.var in K:
            return K[e.var]
        return None
    if isinstance(e, Cast) and BITS.get(e.a.ty, 64) <= 32 and e.ty == "i64":
        if e.a.ty == "b":
            return frozenset()
        return frozenset("s") if e.signed else frozenset("u")
    if isinstance(e, Select):
        a, b = _kind(e.a, K), _kind(e.b, K)
        return None if a is None or b is None else a | b
    if isinstance(e, Bin) and e.ty == "i64":
        a, b = _kind(e.a, K), _kind(e.b, K)
        if e.op == "&" and (a == frozenset() or b == frozenset()) and \
                (isinstance(e.b, Const) or isinstance(e.a, Const)):
            return frozenset()
        if a is None or b is None:
            return None
        k = a | b
        if e.op in ("*", "+", "-") and not isinstance(e.a, Const) and not isinstance(e.b, Const) and \
                not (isinstance(e.a, Var) and e.a.var in K and isinstance(e.b, Var) and e.b.var in K):
            # (long)a * b, s += (long)m[i][j]: lcc does int arithmetic in 32-bit
            # operations, a 64-bit one on sign-extended values is meant to be long
            return None
        if e.op in ("+", "-", "*") and "u" not in k:
            return frozenset("s")
        if e.op in ("&", "|", "^") and len(k) <= 1:
            return k
        if e.op in (">>s", "/s", "%s") and "u" not in k:
            return frozenset("s")
        if e.op in (">>u", "/u", "%u") and "s" not in k:
            return frozenset("u") if k else frozenset()
        return None
    if isinstance(e, Un) and e.ty == "i64" and e.op in ("-", "~"):
        a = _kind(e.a, K)
        return frozenset("s") if a is not None and "u" not in a else None
    return None


def _truncated(F, v):
    """does the code read v only through 32-bit views somewhere?"""
    for b in F.block_list():
        for st in b.stmts:
            for e in st.exprs():
                for x in e.walk():
                    if isinstance(x, Cast) and isinstance(x.a, Var) and x.a.var is v and \
                            BITS.get(x.ty, 64) <= 32:
                        return True
        for e in (b.term.exprs() if b.term is not None else []):
            for x in e.walk():
                if isinstance(x, Cast) and isinstance(x.a, Var) and x.a.var is v and BITS.get(x.ty, 64) <= 32:
                    return True
    return False


def demote_ints(F):
    from .analysis import collect, simplify_all
    from .ir import map_expr
    defs, uses = collect(F)
    addr = set()                 # used as a memory address: never an int
    for b in F.block_list():
        for st in b.stmts:
            exprs = list(st.exprs()) + ([st.dst] if isinstance(st, Assign) and not isinstance(st.dst, Var) else [])
            if isinstance(st, Store):
                exprs.append(st.addr)
            for e in exprs:
                for x in e.walk():
                    if isinstance(x, Load):
                        addr |= {y.var for y in x.addr.walk() if isinstance(y, Var)}
            if isinstance(st, Store):
                addr |= {y.var for y in st.addr.walk() if isinstance(y, Var)}
    cand = {}
    for v in F.vars:
        if v.ty != "i64" or getattr(v, "pointer", False) or v in addr or v.reg in ("sp", "lc", "pc") or \
                v.reg.startswith(("lsr_", "aa", "apb", "stack")):
            continue
        if v.is_param or v.is_input:
            if not v.is_param or not _truncated(F, v):
                continue
            cand[v] = frozenset("s")       # entry value: an int argument
        elif not defs.get(v):
            continue
        else:
            cand[v] = frozenset()
    # variables whose every read keeps only the low 32 bits (truncation, a
    # 32-bit store, the returned value): any mix of 32-bit values is fine
    low_only = set(cand)
    from .ir import Return

    def scan_uses(e, parent, whole_ok):
        if isinstance(e, Var) and e.var in low_only:
            ok = (isinstance(parent, Cast) and BITS.get(parent.ty, 64) <= 32) or whole_ok
            if not ok:
                low_only.discard(e.var)
            return
        for c in e.children():
            scan_uses(c, e, False)
    for b in F.block_list():
        for st in b.stmts:
            for e in st.exprs():
                whole = isinstance(st, Store) and e is st.val and BITS.get(st.ty, 64) <= 32
                scan_uses(e, None, whole)
        if b.term is not None:
            for e in b.term.exprs():
                scan_uses(e, None, isinstance(b.term, Return))
    K = dict(cand)
    changed = True
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
        for v in list(K):
            k = K[v]
            for b, i in defs.get(v, []):
                st = b.stmts[i]
                d = _kind(st.src, K)
                if d is None:
                    k = None
                    break
                k = k | d
            if k is not None and len(k) == 2 and v in low_only:
                k = frozenset("s")
            if k is None or len(k) == 2:
                del K[v]
                changed = True
            elif k != K[v]:
                K[v] = k
                changed = True
    if not K:
        return {}
    for v, k in K.items():
        v.ty = "i32"
        v.unsigned = k == frozenset("u")

    def fix(e):
        def f(x):
            if isinstance(x, Var) and x.var in K:
                if x.ty == "i64":
                    return Cast(Var(x.var, "i32"), "i64", signed=not x.var.unsigned)
                return Var(x.var, x.ty if x.ty != "i64" else "i32")
            return x
        return map_expr(e, f)
    for b in F.block_list():
        for st in b.stmts:
            st.map(fix)
            if isinstance(st, Assign) and isinstance(st.dst, Var) and st.dst.var in K:
                st.dst = Var(st.dst.var, "i32")
                if st.src.ty != "i32":
                    st.src = Cast(st.src, "i32", signed=True)
            elif isinstance(st, Assign) and not isinstance(st.dst, Var):
                st.dst = fix(st.dst)
        if b.term is not None:
            b.term.map(fix)
    simplify_all(F)
    return K


def ret_kind(vals):
    """common 32-bit kind of the returned values, or None"""
    k = frozenset()
    for v in vals:
        d = _kind(v, {})
        if d is None:
            return None
        k |= d
    return None if len(k) == 2 else k


# ------------------------------------------------------------ signedness
U_OPS = {">>u", "/u", "%u", "<u", "<=u", ">u", ">=u"}
S_OPS = {">>s", "/s", "%s", "<s", "<=s", ">s", ">=s"}


def _sign_votes(e, ucount, scount, weak=None):
    """votes of the variables read directly by unsigned / signed operations"""
    def var_of(x):
        while isinstance(x, Cast) and BITS.get(x.ty, 64) == BITS.get(x.a.ty, 64):
            x = x.a
        return x.var if isinstance(x, Var) else None

    CMPS = {"<u", "<=u", ">u", ">=u"}
    weak_votes = weak

    def visit(x):
        if isinstance(x, Bin) and (x.op in U_OPS or x.op in S_OPS):
            ops = (x.a,) if x.op in (">>u", ">>s") else (x.a, x.b)
            # `x <u K`: a switch / range check, says nothing about x's type
            weak = x.op in CMPS and (isinstance(x.a, Const) or isinstance(x.b, Const))
            shift_k = x.op == ">>u" and isinstance(x.b, Const)
            for o in ops:
                v = var_of(o)
                if v is not None and not weak:
                    if shift_k and weak_votes is not None:
                        weak_votes[v] = weak_votes.get(v, 0) + 1   # n >> 1 for a known n >= 0 too
                        continue
                    d = ucount if x.op in U_OPS else scount
                    d[v] = d.get(v, 0) + 1
        if isinstance(x, Cast) and BITS.get(x.ty, 64) > BITS.get(x.a.ty, 64) and x.a.ty != "b" and x.signed:
            # sign extension: an int (zero extension is also used for ints known >= 0)
            v = var_of(x.a)
            if v is not None:
                scount[v] = scount.get(v, 0) + 1
        for c in x.children():
            visit(c)
    visit(e)


def expr_unsigned(e):
    """does e compute an unsigned value (unsigned op on top of +-*&|^ chains,
    unsigned variables, no signed ones)?"""
    u = s = 0
    stack = [e]
    while stack:
        x = stack.pop()
        if isinstance(x, Var):
            if getattr(x.var, "unsigned", False):
                u += 1
            elif x.ty in ("i32", "i64") and not getattr(x.var, "pointer", False):
                s += 1
            continue
        if isinstance(x, Bin):
            if x.op in U_OPS:
                u += 1
                continue
            if x.op in S_OPS:
                s += 1
                continue
            if x.op in ("+", "-", "*", "&", "|", "^", "<<"):
                stack.extend([x.a, x.b])
            continue
        if isinstance(x, Cast) and BITS.get(x.ty, 64) <= BITS.get(x.a.ty, 64):
            stack.append(x.a)
            continue
        if isinstance(x, Cast):
            if x.signed:
                s += 1
            else:
                u += 1
    return u > 0 and s == 0


def signedness(F):
    """unsigned int variables: read by unsigned operations only"""
    ucount, scount, wcount = {}, {}, {}
    defs_u = {}
    for b in F.block_list():
        for st in b.stmts:
            for e in st.exprs():
                _sign_votes(e, ucount, scount, wcount)
            if isinstance(st, Assign) and isinstance(st.dst, Var):
                src = st.src
                if isinstance(src, Bin) and src.op in ("/u", "%u", ">>u"):
                    defs_u[st.dst.var] = defs_u.get(st.dst.var, 0) + 1
        for e in (b.term.exprs() if b.term is not None else []):
            _sign_votes(e, ucount, scount, wcount)
    for v in F.vars:
        if v.ty not in ("i32", "i64", "i16", "i8") or getattr(v, "pointer", False) or v.reg in ("lc", "pc"):
            continue
        weak = wcount.get(v, 0) if v.is_param else 0    # locals: n >> 1 of a positive count
        if (ucount.get(v, 0) or defs_u.get(v, 0) or weak) and not scount.get(v, 0):
            v.unsigned = True


# ------------------------------------------------------------ narrow params, function pointers
def narrow_params(F):
    """char / short parameters: every read truncates to the same narrow width."""
    from .ir import map_expr
    for v in F.vars:
        if not v.is_param or getattr(v, "pointer", False) or v.ty not in ("i32", "i64"):
            continue
        widths, uns, ok = set(), [], True
        for b in F.block_list():
            items = []
            for st in b.stmts:
                if isinstance(st, Assign) and isinstance(st.dst, Var) and st.dst.var is v:
                    ok = False
                for e in st.exprs():
                    narrow_store = isinstance(st, Store) and e is st.val and BITS.get(st.ty, 64) <= 16
                    items.append((e, st.ty if narrow_store else None))
            for e in (b.term.exprs() if b.term is not None else []):
                items.append((e, None))
            for e, store_w in items:
                def visit(x, parent):
                    nonlocal ok
                    if isinstance(x, Var) and x.var is v:
                        if isinstance(parent, Cast) and parent.ty in ("i8", "i16"):
                            widths.add(parent.ty)
                            uns.append(not parent.signed)
                        elif parent is None and store_w:
                            widths.add(store_w)
                        else:
                            ok = False
                        return
                    for c in x.children():
                        visit(c, x)
                visit(e, None)
        if not ok or len(widths) != 1:
            continue
        w = widths.pop()
        v.ty = w
        v.unsigned = w == "i16" and bool(uns) and all(uns)

        def fix(x, v=v, w=w):
            if isinstance(x, Cast) and x.ty == w and isinstance(x.a, Var) and x.a.var is v:
                return Var(v, w)
            if isinstance(x, Var) and x.var is v:
                return Var(v, w)
            return x
        for b in F.block_list():
            for st in b.stmts:
                st.map(lambda e: map_expr(e, fix))
            if b.term is not None:
                b.term.map(lambda e: map_expr(e, fix))


def func_pointers(F):
    """variables called through: `int (*a1)()`"""
    from .ir import Call
    for b in F.block_list():
        for st in b.stmts:
            for e in st.exprs():
                for x in e.walk():
                    if isinstance(x, Call) and isinstance(x.target, Var):
                        x.target.var.funcptr = x.ty if x.ret_used else "i64"
        for e in (b.term.exprs() if b.term is not None else []):
            for x in e.walk():
                if isinstance(x, Call) and isinstance(x.target, Var):
                    x.target.var.funcptr = x.ty if x.ret_used else "i64"


# ------------------------------------------------------------ floats kept in integer registers
#
# e2k passes and returns float values in the general registers, so a float
# argument arrives as an int that is only ever reinterpreted:
# __bitcast_float(a2).  Such variables become float; loads and stores whose
# value is only reinterpreted become float accesses (-> float *p).

def fold_float_mem(e):
    """BitCast(Load(a, i32), f32) -> Load(a, f32) (bottom-up)"""
    from .ir import BitCast, map_expr

    def f(x):
        if isinstance(x, BitCast) and is_float(x.ty) and isinstance(x.a, Load) and \
                BITS.get(x.a.ty) == BITS[x.ty] and not is_float(x.a.ty):
            return Load(x.a.addr, x.ty, spec=getattr(x.a, "spec", False))
        if isinstance(x, BitCast) and isinstance(x.a, BitCast) and x.a.a.ty == x.ty:
            return x.a.a
        return x
    return map_expr(e, f)


def fold_float_store(st):
    from .ir import BitCast
    if isinstance(st, Store) and isinstance(st.val, BitCast) and is_float(st.val.a.ty) and \
            BITS.get(st.ty) == BITS[st.val.a.ty]:
        st.val, st.ty = st.val.a, st.val.a.ty


def float_vars(F):
    from .ir import BitCast, map_expr
    from .analysis import collect
    defs, _ = collect(F)
    fuses, bad = {}, set()
    for b in F.block_list():
        exprs = [e for st in b.stmts for e in st.exprs()] + \
                (list(b.term.exprs()) if b.term is not None else [])
        for e in exprs:
            def visit(x, parent):
                if isinstance(x, Var) and x.ty == "i32" and x.var.ty == "i32":
                    if isinstance(parent, BitCast) and parent.ty == "f32":
                        fuses[x.var] = fuses.get(x.var, 0) + 1
                    elif isinstance(parent, Cast) and not (isinstance(parent, Cast) and parent.ty == "i64"
                                                           and not parent.signed):
                        bad.add(x.var)
                    elif not isinstance(parent, Cast):
                        bad.add(x.var)
                    return
                for c in x.children():
                    visit(c, x)
            visit(e, None)
    out = set()
    for v, n in fuses.items():
        if v in bad or getattr(v, "pointer", False):
            continue
        ds = defs.get(v, [])
        if any(not (isinstance(b.stmts[i].src, BitCast) and b.stmts[i].src.a.ty == "f32") or
               b.stmts[i].guard is not None for b, i in ds):
            continue
        if not ds and not v.is_param:
            continue
        out.add(v)
    if not out:
        return out
    for v in out:
        v.ty = "f32"

    def fix(x):
        if isinstance(x, Var) and x.var in out:
            return BitCast(Var(x.var, "f32"), "i32")
        return x
    for b in F.block_list():
        for st in b.stmts:
            st.map(lambda e: map_expr(map_expr(e, fix), lambda y: y))
            if isinstance(st, Assign) and isinstance(st.dst, Var) and st.dst.var in out:
                st.dst = Var(st.dst.var, "f32")
                if isinstance(st.src, BitCast) and st.src.ty == "i32":
                    st.src = st.src.a
            st.map(fold_float_mem)
        if b.term is not None:
            b.term.map(lambda e: fold_float_mem(map_expr(e, fix)))
    return out


# ------------------------------------------------------------ library calls
def apply_libc(F, ctx=None):
    """Known C library prototypes: trim spurious arguments (format strings
    give the count of printf/scanf arguments), mark pointer arguments,
    drop the result of void functions.  Returns {var: pointee hint}."""
    from .ir import Call, Sym, Str
    from . import libc
    hints = {}

    def var_of(a):
        while isinstance(a, Cast):
            a = a.a
        return a.var if isinstance(a, Var) else None

    def visit(x):
        if isinstance(x, Call) and isinstance(x.target, Sym) and x.target.kind == "func":
            p = libc.lookup(x.target.name)
            if p is None:
                return
            decl, ret, params, fmt = p
            n, kinds = len(params), list(params)
            if fmt is not None:
                text = None
                if fmt < len(x.args):
                    a = x.args[fmt]
                    if isinstance(a, Str):
                        text = a.text
                    elif isinstance(a, Sym) and ctx is not None and a.kind == "data":
                        text = ctx.string_at(a)
                if text is not None:
                    r = libc.format_args(text)
                    if r is None:
                        n = None
                    else:
                        n += r[0]
                        scan = "scanf" in x.target.name
                        kinds += ["ptr" if scan and k == "num" else k for k in r[1]]
                else:
                    n = None
            if n is not None and len(x.args) > n:
                x.args = x.args[:n]
            for a, k in zip(x.args, kinds):
                v = var_of(a)
                if v is not None and k in ("str", "ptr") and v.ty == "i64":
                    hints.setdefault(v, "i8" if k == "str" else None)
            if ret == "void":
                x.ret_used = False
    for b in F.block_list():
        for st in b.stmts:
            for e in st.exprs():
                for x in e.walk():
                    visit(x)
        for e in (b.term.exprs() if b.term is not None else []):
            for x in e.walk():
                visit(x)
    return hints


def returns_32(F):
    """pass 1: do all returned values look like 32-bit ints?  Calls to the
    function itself count as 32-bit (recursion)."""
    from .ir import Return
    name = F.func.name
    saved = RET32.get(name)
    RET32[name] = True
    vals = [b.term.value for b in F.block_list() if isinstance(b.term, Return) and b.term.value is not None]
    ok = bool(vals) and ret_kind(vals) is not None
    if saved is None:
        del RET32[name]
    else:
        RET32[name] = saved
    return ok
