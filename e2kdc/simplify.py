"""Algebraic simplification of IR expressions."""
from .ir import (Const, FConst, Bin, Un, Cast, BitCast, Select, Sym, Load, Var, Reg, Call, Intr,
                 map_expr, mask, sext, BITS, is_int, is_float)

SWAP = {"<s": ">s", "<=s": ">=s", ">s": "<s", ">=s": "<=s", "<u": ">u", "<=u": ">=u",
        ">u": "<u", ">=u": "<=u", "==": "==", "!=": "!="}
NEG = {"<s": ">=s", "<=s": ">s", ">s": "<=s", ">=s": "<s", "<u": ">=u", "<=u": ">u",
       ">u": "<=u", ">=u": "<u", "==": "!=", "!=": "=="}
COMMUT = {"+", "*", "&", "|", "^", "==", "!=", "&&", "||"}
TRUNC_OK = {"+", "-", "*", "&", "|", "^", "<<"}


def cval(e):
    return e.value if isinstance(e, Const) else None


def fold_bin(op, a, b, ty):
    bits = BITS.get(ty, 64)
    m = mask(bits) if is_int(ty) else None
    abits = BITS.get(a.ty, 64)
    x, y = a.value, b.value
    xs, ys = sext(x, abits), sext(y, abits)
    try:
        if op == "+": return Const(x + y, ty)
        if op == "-": return Const(x - y, ty)
        if op == "*": return Const(x * y, ty)
        if op == "&": return Const(x & y, ty)
        if op == "|": return Const(x | y, ty)
        if op == "^": return Const(x ^ y, ty)
        if op == "<<": return Const(x << (y & (bits - 1)), ty)
        if op == ">>u": return Const((x & m) >> (y & (bits - 1)), ty)
        if op == ">>s": return Const(sext(x, bits) >> (y & (bits - 1)), ty)
        if op in ("rotr", "rotl") and is_int(ty):
            k = (y % bits) if op == "rotr" else (-y % bits)
            v = x & m
            return Const(((v >> k) | (v << (bits - k))) & m, ty)
        if op == "/u" and y: return Const(x // y, ty)
        if op == "%u" and y: return Const(x % y, ty)
        if op == "/s" and ys:
            q = abs(xs) // abs(ys)
            return Const(q if (xs < 0) == (ys < 0) else -q, ty)
        if op == "==": return Const(int(x == y), "b")
        if op == "!=": return Const(int(x != y), "b")
        if op == "<s": return Const(int(xs < ys), "b")
        if op == "<=s": return Const(int(xs <= ys), "b")
        if op == ">s": return Const(int(xs > ys), "b")
        if op == ">=s": return Const(int(xs >= ys), "b")
        if op == "<u": return Const(int(x < y), "b")
        if op == "<=u": return Const(int(x <= y), "b")
        if op == ">u": return Const(int(x > y), "b")
        if op == ">=u": return Const(int(x >= y), "b")
    except Exception:
        return None
    return None


def strip_ext(e, ty):
    """If e is an integer extension of a value of type ty, return that value."""
    if isinstance(e, Cast) and e.a.ty == ty and is_int(e.ty) and BITS[e.ty] > BITS[ty]:
        return e.a
    if isinstance(e, Const) and is_int(ty):
        return Const(e.value, ty)
    return None


def getf_expr(src, v, ty):
    """getfs/getfd with field spec v: shift v[5:0], size v[11:6], sign v[12]."""
    bits = BITS[ty]
    sh = v & (bits - 1)
    size = (v >> 6) & (bits - 1)
    sgn = bool(v & (1 << 12))
    if size == 0:
        return Const(0, ty)
    # qemu: rotate (not shift) the source right by `shift`, then take the low
    # `size` bits: a field may wrap around the top (shift=31, size=17 is
    # bit 31 followed by bits 0..15, i.e. 2*x for a 16-bit x)
    if sh + size > bits:
        x = Bin("rotr", src, Const(sh, ty), ty)
    else:
        x = Bin(">>u", src, Const(sh, ty), ty) if sh else src
    if size in (8, 16, 32) and size < bits:
        return Cast(Cast(x, "i%d" % size, signed=False), ty, signed=sgn)
    if sgn:
        return Bin(">>s", Bin("<<", x, Const(bits - size, ty), ty), Const(bits - size, ty), ty)
    return Bin("&", x, Const((1 << size) - 1, ty), ty)


def _sum_terms(e, sign, out):
    if isinstance(e, Bin) and e.ty == "i64" and e.op in ("+", "-"):
        _sum_terms(e.a, sign, out)
        _sum_terms(e.b, sign if e.op == "+" else -sign, out)
    else:
        out.append([sign, e])


def cancel_syms(e):
    """(&g + 64 + p) - &g - 64  ->  p : addresses of one symbol cancel in a sum
    (a pointer induction variable rebased against the array it walks)."""
    terms = []
    _sum_terms(e, 1, terms)
    if len(terms) < 2 or all(t[0] > 0 for t in terms):
        return None
    hit = False
    k = 0
    for t in terms:
        if t[1] is None or isinstance(t[1], Const):
            continue
        for u in terms:
            if u is t or u[1] is None or u[0] != -t[0]:
                continue
            if isinstance(t[1], Sym) and isinstance(u[1], Sym) and u[1].name == t[1].name:
                k += t[0] * t[1].off + u[0] * u[1].off
            elif not isinstance(t[1], Sym) and not isinstance(u[1], Const) and \
                    getattr(t[1], "pure", True) and repr_key(t[1]) == repr_key(u[1]):
                pass                              # (p + 2*i + 4) - (p + 4): p cancels
            else:
                continue
            t[1] = u[1] = None
            hit = True
            break
    if not hit:
        return None
    rest = [t for t in terms if t[1] is not None]
    for t in rest:
        if isinstance(t[1], Const):
            k += t[0] * sext(t[1].value, 64)
    rest = [t for t in rest if not isinstance(t[1], Const)]
    pos = [t[1] for t in rest if t[0] > 0]
    neg = [t[1] for t in rest if t[0] < 0]
    r = pos[0] if pos else Const(0, "i64")
    for x in pos[1:]:
        r = Bin("+", r, x, "i64")
    for x in neg:
        r = Bin("-", r, x, "i64")
    if k:
        r = Bin("+", r, Const(k, "i64"), "i64")
    return r


def _zext_width(e):
    """bits a non-negative value e fits in, if structurally known"""
    if isinstance(e, Cast) and not e.signed and is_int(e.a.ty) and is_int(e.ty) and BITS[e.a.ty] < BITS[e.ty]:
        return BITS[e.a.ty]
    if isinstance(e, Const) and is_int(e.ty):
        return max(e.value.bit_length(), 1)
    if isinstance(e, Bin) and e.op == "&" and isinstance(e.b, Const):
        return max(e.b.value.bit_length(), 1)
    return None


def narrow_rotates(roots, map_root):
    """v = (unsigned short)*p; ... rotl(v, 1)  ->  v << 1: a rotation of a value
    narrow enough that nothing wraps around is a shift (lcc computes a*3 of a
    short field with getfs, which rotates)"""
    from .ir import Assign
    width, bad = {}, set()
    for h in roots:
        if isinstance(h, Assign) and isinstance(h.dst, Var):
            v = h.dst.var
            w = _zext_width(h.src) if h.guard is None else None
            if w is None:
                bad.add(v)
            else:
                width[v] = max(width.get(v, 0), w)
    width = {v: w for v, w in width.items() if v not in bad and not v.is_param and not v.is_input}
    if not width:
        return

    def f(x):
        if isinstance(x, Bin) and x.op in ("rotr", "rotl") and isinstance(x.b, Const) and is_int(x.ty) \
                and isinstance(x.a, Var) and x.a.var in width:
            bits = BITS[x.ty]
            n = (bits - x.b.value) % bits if x.op == "rotr" else x.b.value % bits
            if n and width[x.a.var] + n <= bits:
                return Bin("<<", x.a, Const(n, x.ty), x.ty)
        return x
    for h in roots:
        map_root(h, lambda e: simplify(map_expr(e, f)))


def simp(e):
    if isinstance(e, Bin) and e.op in ("<<", ">>s") and isinstance(e.b, Const) and is_int(e.ty) \
            and isinstance(e.a, Bin) and e.a.op == "<<" and isinstance(e.a.b, Const):
        bits, a, b = BITS[e.ty], e.a.b.value, e.b.value
        if e.op == "<<" and a + b < bits:
            return simp(Bin("<<", e.a.a, Const(a + b, e.ty), e.ty))       # (x << a) << b
        if e.op == ">>s" and a >= b and a < bits and bits - a in (8, 16, 32) and bits - a < bits:
            # (x << 16) >> 15 on int: sign extension of the low half, then << 1
            n = {8: "i8", 16: "i16", 32: "i32"}[bits - a]
            sx = Cast(Cast(e.a.a, n, signed=True), e.ty, signed=True)
            return simp(Bin("<<", sx, Const(a - b, e.ty), e.ty)) if a > b else sx
    if isinstance(e, Bin) and e.op in ("+", "-") and e.ty == "i64":
        r = cancel_syms(e)
        if r is not None:
            return simp(r)
    if isinstance(e, Bin) and e.op in ("rotr", "rotl") and isinstance(e.b, Const) and is_int(e.ty):
        bits = BITS[e.ty]
        n = (bits - e.b.value) % bits if e.op == "rotr" else e.b.value % bits
        a = e.a
        if n and isinstance(a, Cast) and not a.signed and is_int(a.a.ty) and BITS[a.a.ty] + n <= bits:
            return simp(Bin("<<", a, Const(n, e.ty), e.ty))   # nothing wraps: a plain shift
    if isinstance(e, Bin) and e.op == "&" and is_int(e.ty):
        for x, y in ((e.a, e.b), (e.b, e.a)):
            if isinstance(y, Select) and isinstance(y.a, Const) and isinstance(y.b, Const):
                # a & (c ? K1 : K2)  ->  c ? a & K1 : a & K2   (masks from packed compares)
                return simp(Select(y.c, simp(Bin("&", x, y.a, e.ty)), simp(Bin("&", x, y.b, e.ty)), e.ty))
    if isinstance(e, Intr) and e.name in ("getfs", "getfd") and len(e.args) == 2 and cval(e.args[1]) is not None:
        return simp(getf_expr(e.args[0], cval(e.args[1]), e.ty))   # field spec became a constant
    if isinstance(e, Intr) and e.name in ("paddw", "psubw") and len(e.args) == 2 and cval(e.args[1]) == 0:
        return e.args[0]                              # x + {0, 0}
    if isinstance(e, Bin) and e.op in ("==", "!=") and isinstance(e.a, Cast) and not e.a.signed and \
            isinstance(e.b, Const) and is_int(e.a.ty) and is_int(e.a.a.ty) and \
            BITS[e.a.a.ty] < BITS[e.a.ty] and 0 <= e.b.value < (1 << (BITS[e.a.a.ty] - 1)):
        # zext(x) == small K  <=>  sext(x) == K: print the plain (signed) element
        return Bin(e.op, Cast(e.a.a, e.a.ty, signed=True), e.b, e.ty)
    if isinstance(e, Intr) and e.name == "st_enbl" and len(e.args) == 6:
        # loop-mode store enable: semc || pcnt == 0 && !(vlc && lcnt < 2 && strmd <= op)
        semc, pc, vlc, lc, strmd, op = [cval(x) for x in e.args]
        if semc is not None and semc:
            return Const(1, "b")
        if semc == 0 and pc is not None:
            if pc:
                return Const(0, "b")
            if vlc == 0 or (strmd is not None and op is not None and strmd > op):
                return Const(1, "b")
            if vlc and strmd is not None and op is not None:
                return simp(Bin(">u", e.args[3], Const(1, e.args[3].ty), "b"))
    if isinstance(e, Intr) and e.name == "pshufb" and len(e.args) == 3 and cval(e.args[2]) is not None:
        # all eight control bytes pick the same plain byte n -> byte broadcast
        cs = {(cval(e.args[2]) >> (8 * i)) & 0xff for i in range(8)}
        if len(cs) == 1 and not cs.copy().pop() & 0xf0:
            n = cs.pop()
            src = e.args[1] if n < 8 else e.args[0]
            sh = 8 * (n & 7)
            x = Cast(src, "i64", signed=False) if src.ty != "i64" else src
            if sh:
                x = Bin(">>u", x, Const(sh, "i64"), "i64")
            byte = simp(Bin("&", x, Const(0xff, "i64"), "i64"))
            return simp(Bin("*", byte, Const(0x0101010101010101, "i64"), "i64"))
    if isinstance(e, Bin):
        a, b, op, ty = e.a, e.b, e.op, e.ty
        if is_int(ty) or ty == "b":
            if isinstance(a, Const) and isinstance(b, Const) and is_int(a.ty):
                r = fold_bin(op, a, b, ty)
                if r is not None:
                    return r
            # canonical: constant on the right
            if isinstance(a, Const) and not isinstance(b, Const):
                if op in COMMUT:
                    return simp(Bin(op, b, a, ty))
                if op in SWAP:
                    return simp(Bin(SWAP[op], b, a, ty))
            bv = cval(b)
            if bv is not None and is_int(ty) and op in ("&", ">>u") and isinstance(a, Bin) \
                    and a.op == "|" and (isinstance(a.a, Const) or isinstance(a.b, Const)):
                # (c | x) & m,  (c | x) >> k : distribute so the constant folds (LSR fields)
                return simp(Bin("|", simp(Bin(op, a.a, b, ty)), simp(Bin(op, a.b, b, ty)), ty))
            if bv is not None and is_int(ty) and isinstance(a, Cast) and is_int(a.a.ty) \
                    and BITS[a.a.ty] < BITS[ty] and not a.signed:
                w = BITS[a.a.ty]
                if op == "&" and bv & mask(w) == mask(w):
                    return a                               # zext(x) & 0xffffffff
                if op == ">>u" and bv >= w:
                    return Const(0, ty)                    # zext(x) >> 32
            if bv is not None and is_int(ty) and op == "&" and isinstance(a, Cast) and a.signed \
                    and is_int(a.a.ty) and BITS[a.a.ty] < BITS[ty] and bv == mask(BITS[a.a.ty]):
                return Cast(a.a, ty, signed=False)         # sext(x) & 0xffffffff -> zext(x)
            if bv is not None and is_int(ty) and op == "&" and isinstance(a, Bin) and a.op == "rotr" \
                    and isinstance(a.b, Const):
                # rotr(x, k) & M with M inside the top k bits  ->  (x << (w-k)) & M
                w = BITS[ty]
                k = a.b.value % w
                if k and bv & mask(w - k) == 0:
                    return simp(Bin("&", Bin("<<", a.a, Const(w - k, ty), ty), b, ty))
            if bv is not None and is_int(ty) and op == "&" and isinstance(a, Bin) and a.op == "<<" \
                    and isinstance(a.b, Const) and (mask(BITS[ty]) << a.b.value) & mask(BITS[ty]) == bv:
                return a                                   # (x << k) & (~0 << k)
            if bv is not None and is_int(ty):
                if bv == 0 and op in ("+", "-", "|", "^", "<<", ">>u", ">>s"):
                    return a
                if bv == 1 and op in ("*", "/s", "/u"):
                    return a
                if bv == 0 and op in ("*", "&"):
                    return Const(0, ty)
                if bv == mask(BITS[ty]) and op == "&":
                    return a
                if op == "-":
                    return simp(Bin("+", a, Const(-sext(bv, BITS[ty]), ty), ty))
                if op == "+" and isinstance(a, Bin) and a.op == "+" and isinstance(a.b, Const):
                    return simp(Bin("+", a.a, Const(a.b.value + bv, ty), ty))
                if op == "+" and isinstance(a, Sym):
                    return Sym(a.name, a.off + sext(bv, 64), a.kind)
                if op == "*" and bv & (bv - 1) == 0 and bv > 1:
                    pass
            if op == "+" and is_int(ty):
                for sh, other in ((a, b), (b, a)):
                    if isinstance(sh, Bin) and sh.op == "<<" and isinstance(sh.b, Const) \
                            and sh.b.value < 16 and value_key(sh.a) == value_key(other):
                        return Bin("*", other, Const((1 << sh.b.value) + 1, ty), ty)
            if op == "*" and isinstance(a, Bin) and a.op == "<<" and isinstance(a.b, Const) and bv is not None:
                return simp(Bin("*", a.a, Const(bv << a.b.value, ty), ty))
            av = cval(a)
            if av == 0 and op == "-" and is_int(ty):
                return Un("-", b, ty)
            # (x + c) + y -> (x + y) + c
            if op == "+" and isinstance(a, Bin) and a.op == "+" and isinstance(a.b, Const) \
                    and not isinstance(b, Const):
                return simp(Bin("+", Bin("+", a.a, b, ty), a.b, ty))
            # comparisons of ext'ed values
            if op in NEG and ty == "b" and is_int(a.ty):
                pass
            if op in ("==", "!=") and a.ty == "b" and cval(b) is not None:
                if (cval(b) == 0) == (op == "=="):
                    return simp(Un("!", a, "b"))
                return a
        if op in ("&&", "||", "&", "|") and repr_key(a) == repr_key(b):
            return a                                     # x && x, x | x
        if op == "&&":
            if cval(a) == 1: return b
            if cval(b) == 1: return a
        if op == "||":
            if cval(a) == 0: return b
            if cval(b) == 0: return a
        return e
    if isinstance(e, Un) and e.op == "~" and isinstance(e.a, Select) and isinstance(e.a.a, Const) \
            and isinstance(e.a.b, Const) and is_int(e.ty):
        m = mask(BITS[e.ty])
        return Select(e.a.c, Const(~e.a.a.value & m, e.ty), Const(~e.a.b.value & m, e.ty), e.ty)
    if isinstance(e, Un):
        a = e.a
        if e.op == "!":
            if isinstance(a, Un) and a.op == "!":
                return a.a
            if isinstance(a, Bin) and a.op in NEG and is_int(a.a.ty):
                return Bin(NEG[a.op], a.a, a.b, "b")
            if isinstance(a, Bin) and a.op == "&&":
                return simp(Bin("||", simp(Un("!", a.a, "b")), simp(Un("!", a.b, "b")), "b"))
            if isinstance(a, Bin) and a.op == "||":
                return simp(Bin("&&", simp(Un("!", a.a, "b")), simp(Un("!", a.b, "b")), "b"))
            if isinstance(a, Const):
                return Const(int(not a.value), "b")
        if e.op == "-" and isinstance(a, Const) and is_int(e.ty):
            return Const(-a.value, e.ty)
        if e.op == "~" and isinstance(a, Const) and is_int(e.ty):
            return Const(~a.value, e.ty)
        return e
    if isinstance(e, Cast):
        a = e.a
        if a.ty == e.ty and is_int(e.ty):
            return a
        if isinstance(a, Const) and is_int(e.ty) and is_int(a.ty):
            v = sext(a.value, BITS[a.ty]) if e.signed else a.value & mask(BITS[a.ty])
            return Const(v, e.ty)
        if isinstance(a, Const) and is_float(e.ty) and is_int(a.ty):
            v = sext(a.value, BITS[a.ty]) if e.signed else a.value
            return FConst(float(v), e.ty)
        if is_int(e.ty) and is_int(a.ty):
            tb, ab = BITS[e.ty], BITS[a.ty]
            if isinstance(a, Call) and tb < ab:
                a.ty = e.ty
                return a
            if isinstance(a, Cast) and is_int(a.a.ty):
                ib = BITS[a.a.ty]
                # extension then truncation to a width still >= source: one extension
                if ib < tb <= ab:
                    return Cast(a.a, e.ty, a.signed)
                # truncate an extension
                if tb <= ab and ib >= tb:
                    return simp(Cast(a.a, e.ty, e.signed)) if ib != tb else a.a
                # extend an extension
                if tb > ab and ab > ib:
                    if a.signed == e.signed or not a.signed:
                        return Cast(a.a, e.ty, a.signed)
            # truncation distributes over + - * & | ^ <<
            if tb < ab and isinstance(a, Bin) and a.op in TRUNC_OK:
                lhs = simp(Cast(a.a, e.ty, e.signed))
                rhs = simp(Cast(a.b, e.ty, e.signed))
                return simp(Bin(a.op, lhs, rhs, e.ty))
            if tb < ab and isinstance(a, Select):
                return simp(Select(a.c, simp(Cast(a.a, e.ty, e.signed)), simp(Cast(a.b, e.ty, e.signed)), e.ty))
        return e
    if isinstance(e, BitCast):
        if e.a.ty == e.ty:
            return e.a
        return e
    if isinstance(e, Select):
        if e.a == e.b:
            return e.a
        if isinstance(e.c, Un) and e.c.op == "!":
            return Select(e.c.a, e.b, e.a, e.ty)
        if isinstance(e.c, Const):
            return e.a if e.c.value else e.b
        # select(c, 1, 0) on bools
        if e.ty == "b" and cval(e.a) == 1 and cval(e.b) == 0:
            return e.c
        return e
    return e


_rewrote = [False]


def _simp_seen(x):
    y = simp(x)
    if y is not x:
        _rewrote[0] = True
    return y


def simplify(e):
    """simp bottom-up to a fixpoint (at most 6 passes).  A pass in which simp
    rewrote nothing returned a copy of its input, and simp only looks at the
    tree: the next pass could not change it either, so stop there (the key
    comparison alone never stops a tree with a call: Call keys are ids)"""
    prev = None
    for _ in range(6):
        _rewrote[0] = False
        e = map_expr(e, _simp_seen)
        if not _rewrote[0]:
            break
        k = repr_key(e)
        if k == prev:
            break
        prev = k
    return e


def value_key(e):
    """repr_key, except that the signedness of a truncation does not matter:
    (short)(unsigned short)v and (short)v are the same value"""
    def norm(x):
        if isinstance(x, Cast) and is_int(x.ty) and is_int(x.a.ty) and BITS[x.ty] < BITS[x.a.ty] and not x.signed:
            return Cast(x.a, x.ty, signed=True)
        return x
    return repr_key(map_expr(e, norm))


def repr_key(e):
    try:
        return e.key() if not isinstance(e, Load) else ("L", repr_key(e.addr), e.ty)
    except Exception:
        return id(e)
