"""Algebraic simplification of IR expressions."""
from .ir import (Const, FConst, Bin, Un, Cast, BitCast, Select, Sym, Load, Var, Reg, Call,
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


def simp(e):
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
                            and sh.b.value < 16 and sh.a == other:
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
        if op == "&&":
            if cval(a) == 1: return b
            if cval(b) == 1: return a
        if op == "||":
            if cval(a) == 0: return b
            if cval(b) == 0: return a
        return e
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


def simplify(e):
    prev = None
    for _ in range(6):
        e = map_expr(e, simp)
        k = repr_key(e)
        if k == prev:
            break
        prev = k
    return e


def repr_key(e):
    try:
        return e.key() if not isinstance(e, Load) else ("L", repr_key(e.addr), e.ty)
    except Exception:
        return id(e)
