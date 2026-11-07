"""Structure recovery from the accesses made through pointer variables.

Pointer variables are grouped into classes (copies, `q = p + index`).  For
each class the loads and stores give (offset, type) pairs:

* a stride K: the pointer is indexed as `p + i*K + c` or stepped `p += K`
  with K larger than the accessed values -> an array of K-byte structures
  (an unrolled loop's stride is reduced to the smallest period whose layout
  repeats, so `r[2*i]`, `r[2*i+1]` of an 8-byte struct are recognised);
* no stride, but fields of different types, a field holding a pointer to the
  same class (`n = n->next`) or an indexed array after a scalar field
  (`b->len`, `b->data[i]`) -> one structure.

Fields are named by offset (`f0`, `f4`, ...); gaps become `char _pN[k]`, so the
printed layout matches the original byte for byte.  Identical layouts share
one name across the file.  Accesses that do not fit a field (a 64-bit vector
load over two int fields) keep an explicit cast.
"""
from math import gcd
from .ir import Var, Bin, Un, Const, Cast, Load, Store, Assign, BITS, sext, is_float

ELEMS = ("i8", "i16", "i32", "i64", "f32", "f64")


class Struct:
    stride = False

    def __init__(self):
        self.fields = {}     # off -> dict(ty=ir elem or "ptr", uns=bool, target=Struct|elem|None, count=n|None)
        self.size = None     # bytes (stride of an array of them), None: unknown
        self.flexible = False
        self.name = None

    def key(self):
        def tk(f):
            t = rep(f["target"])
            tgt = ("S", id(t)) if isinstance(t, Struct) else t
            return (f["ty"], f["uns"], tgt if f["ty"] in ("ptr", "struct") else None, f["count"])
        return (self.size, tuple(sorted((o, tk(f)) for o, f in self.fields.items())))

    def field_at(self, off):
        f = self.fields.get(off)
        return f

    def array_field(self, off):
        """(field offset, field) of an array field containing off"""
        for o, f in self.fields.items():
            if f["count"] is not None and f["ty"] != "ptr":
                s = BITS[f["ty"]] // 8
                end = o + s * f["count"] if f["count"] else None
                if o <= off and (end is None or off < end) and (off - o) % s == 0:
                    return o, f
        return None


def _fold_byte_runs(S, min_run=8):
    """eight or more single-byte fields in a row are a char array (a fully
    unrolled loop over `char data[20]` stores data[0..19] one by one)"""
    offs = sorted(o for o, f in S.fields.items() if f["ty"] == "i8" and f["count"] is None)
    run = []
    for o in offs + [None]:
        if run and o == run[-1] + 1:
            run.append(o)
            continue
        if len(run) >= min_run:
            f = S.fields[run[0]]
            for x in run[1:]:
                del S.fields[x]
            f["count"] = len(run)
        run = [o] if o is not None else []


def fsize(f):
    if f["ty"] == "struct":
        return rep(f["target"]).size or 1
    return 8 if f["ty"] == "ptr" else BITS[f["ty"]] // 8


def span(f):
    return fsize(f) * (f["count"] or 1)


def field_path(S, fo, rest, ty, unsigned, printer):
    """text after `p->` / `obj.` for an access of type ty at byte offset fo
    (+ rest) inside S: f8, f8.f4, f4[k]; None if no field fits"""
    from .types import scale_down
    from .simplify import simplify
    S = rep(S)
    f = S.fields.get(fo)
    if rest is None and f is not None and f["count"] is None and f["ty"] != "struct":
        if f["ty"] == ty or (f["ty"] == "ptr" and ty == "i64"):
            if unsigned and f["ty"] != "ptr" and not f["uns"] and not is_float(ty):
                return None
            return "f%x" % fo, field_ctype(S, f)
        return None
    for o, g in sorted(S.fields.items()):
        if not (o <= fo and (fo < o + span(g) or (g["count"] == 0))):
            continue
        if g["ty"] == "struct" and g["count"] is None:
            sub = field_path(g["target"], fo - o, rest, ty, unsigned, printer)
            return ("f%x.%s" % (o, sub[0]), sub[1]) if sub else None
        if g["count"] is not None and g["ty"] not in ("struct", "ptr"):
            if g["ty"] != ty or (unsigned and not g["uns"] and not is_float(ty)):
                return None
            sz = BITS[ty] // 8
            start = fo - o
            if start % sz:
                return None
            k = Const(start // sz, "i64")
            if rest is not None:
                r = scale_down(rest, sz)
                if r is None:
                    return None
                k = simplify(Bin("+", r, k, r.ty)) if start else simplify(r)
            return "f%x[%s]" % (o, printer.e(k)), field_ctype(S, g)
        return None
    return None


def const_part(e):
    """e -> (rest or None, constant)"""
    if isinstance(e, Const):
        return None, sext(e.value, BITS.get(e.ty, 64))
    if isinstance(e, Bin) and e.op in ("+", "-") and isinstance(e.b, Const):
        r, c = const_part(e.a)
        k = sext(e.b.value, BITS.get(e.b.ty, 64))
        return r, c + (k if e.op == "+" else -k)
    if isinstance(e, Bin) and e.op == "+":
        ra, ca = const_part(e.a)
        rb, cb = const_part(e.b)
        if ra is None:
            return rb, ca + cb
        if rb is None:
            return ra, ca + cb
        return Bin("+", ra, rb, e.ty), ca + cb
    return e, 0


def lin(e):
    """Pull constants and factors out of sign-extended 32-bit indices:
    (long)(2*i + 1) * 4  ->  (long)i * 8 + 4.  Exact unless the int index
    overflows (undefined in the source)."""
    if isinstance(e, Bin) and e.op in ("+", "-"):
        return Bin(e.op, lin(e.a), lin(e.b), e.ty)
    if isinstance(e, Bin) and e.op in ("*", "<<") and isinstance(e.b, Const):
        k = sext(e.b.value, 64) if e.op == "*" else (1 << e.b.value if e.b.value < 32 else None)
        a = e.a
        if k is None:
            return e
        if isinstance(a, Cast) and a.signed and a.ty == "i64" and a.a.ty == "i32":
            x = a.a
            if isinstance(x, Bin) and x.op in ("+", "-") and isinstance(x.b, Const):
                c = sext(x.b.value, 32) * (1 if x.op == "+" else -1)
                inner = lin(Bin("*", Cast(x.a, "i64", signed=True), Const(k, "i64"), "i64"))
                return Bin("+", inner, Const(c * k, "i64"), "i64")
            if isinstance(x, Bin) and x.op == "*" and isinstance(x.b, Const):
                m = sext(x.b.value, 32)
                return lin(Bin("*", Cast(x.a, "i64", signed=True), Const(m * k, "i64"), "i64"))
        la = lin(a)
        if e.ty == "i64":
            # (2*i + 4) * 1  ->  2*i + 4,  (x + c) * k  ->  x*k + c*k  (exact mod 2^64)
            if k == 1:
                return la
            if isinstance(la, Const):
                return Const(sext(la.value, 64) * k, "i64")
            if isinstance(la, Bin) and la.op in ("+", "-"):
                return Bin(la.op, lin(Bin("*", la.a, Const(k, "i64"), "i64")),
                           lin(Bin("*", la.b, Const(k, "i64"), "i64")), "i64")
            if isinstance(la, Bin) and la.op == "*" and isinstance(la.b, Const):
                return Bin("*", la.a, Const(sext(la.b.value, 64) * k, "i64"), "i64")
        if e.op == "<<":
            return Bin("*", la, Const(k, "i64"), "i64")
        return Bin("*", la, e.b, e.ty)
    return e


def stride_of(e):
    """largest K with e = K * (something), structurally; 1 if unknown"""
    if isinstance(e, Bin) and e.op == "*" and isinstance(e.b, Const):
        return (abs(sext(e.b.value, 64)) or 1) * (stride_of(e.a) if not isinstance(e.a, Const) else 1)
    if isinstance(e, Bin) and e.op == "<<" and isinstance(e.b, Const) and e.b.value < 32:
        return (1 << e.b.value) * stride_of(e.a)
    if isinstance(e, Bin) and e.op in ("+", "-"):
        return gcd(stride_of(e.a), stride_of(e.b))
    if isinstance(e, Un) and e.op == "-":
        return stride_of(e.a)
    if isinstance(e, Const):
        return abs(sext(e.value, 64)) or 0
    return 1


def infer(roots, exprs_of):
    """roots: AST statement holders.  Sets VarInfo.struct; returns the list of
    Struct objects created."""
    from .types import split_addr
    ptrs = {}                 # ordered set: VarInfo hashes by address
    parent = {}

    def find(v):
        while parent.get(v, v) is not v:
            parent[v] = parent.get(parent[v], parent[v])
            v = parent[v]
        return v

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra is not rb:
            parent[rb] = ra

    def ptr_var(e):
        return e.var if isinstance(e, Var) and getattr(e.var, "pointer", False) else None

    arrays_of_sym = set()
    accesses = []            # (var, rest, c, ty, uns)
    steps = {}               # var -> {K}
    ptrloads = []            # (base var, rest, c, dst var)

    def access(addr, ty, uns):
        r = split_addr(addr)
        if r is None or ty not in ELEMS:
            return
        v, off = r
        ptrs[v] = True
        rest, c = const_part(lin(off))
        accesses.append((v, rest, c, ty, uns))

    def walk(e, parent_=None):
        if isinstance(e, Cast) and isinstance(e.a, Load) and BITS.get(e.ty, 0) > BITS.get(e.a.ty, 64):
            access(e.a.addr, e.a.ty, not e.signed)
            for c in e.a.children():
                walk(c, e.a)
            return
        if isinstance(e, Load):
            access(e.addr, e.ty, None)
        for c in e.children():
            walk(c, e)

    loadstores = []          # *(p + c) = *(q + d): (p, rest, c, q, d)
    for h in roots:
        if isinstance(h, Store):
            access(h.addr, h.ty, None)
            if h.ty == "i64":
                r = split_addr(h.addr)
                if r is not None:
                    rest, c = const_part(lin(r[1]))
                    val = h.val
                    while isinstance(val, Cast) and BITS.get(val.ty, 64) == 64:
                        val = val.a
                    if ptr_var(val) is not None:
                        ptrloads.append((r[0], rest, c, val.var))      # *(p + c) = w: w's type
                    elif isinstance(val, Load) and val.ty == "i64":
                        q = split_addr(val.addr)
                        if q is not None:
                            qr, qc = const_part(lin(q[1]))
                            if qr is None:
                                loadstores.append((r[0], rest, c, q[0], qc))
        if isinstance(h, Assign) and isinstance(h.dst, Var) and getattr(h.dst.var, "pointer", False):
            d, src = h.dst.var, h.src
            ptrs[d] = True
            from .ir import Sym
            if any(isinstance(x, Sym) and x.kind in ("local", "data") for x in src.walk()):
                arrays_of_sym.add(d)         # points into a known stack / global array
            while isinstance(src, Cast) and BITS.get(src.ty, 64) == 64:
                src = src.a
            if ptr_var(src) is not None:
                union(d, src.var)
            elif isinstance(src, Bin) and src.op in ("+", "-"):
                r = split_addr(src)
                if r is not None:
                    b, off = r
                    rest, c = const_part(off)
                    if b is d and rest is None and c and src.op == "+":
                        steps.setdefault(d, set()).add(abs(c))
                    elif b is not d and rest is not None and c == 0:
                        union(d, b)          # q = p + i*K: same array
            elif isinstance(src, Load) and src.ty == "i64":
                r = split_addr(src.addr)
                if r is not None:
                    rest, c = const_part(r[1])
                    ptrloads.append((r[0], rest, c, d))
        for e in exprs_of(h):
            walk(e)
    # *(p + c) = *(q + d): the slot of p holds what q's slot holds
    for p, rest, c, q, d in loadstores:
        src = next((w for b, r2, c2, w in ptrloads if find(b) is find(q) and r2 is None and c2 == d), None)
        if src is not None:
            ptrloads.append((p, rest, c, src))
    # per class
    classes = {}
    for v in ptrs:
        classes.setdefault(find(v), {})[v] = True
    out = []
    for root, members in classes.items():
        acc = [a for a in accesses if find(a[0]) is root]
        pl = [p for p in ptrloads if find(p[0]) is root]
        if not acc and not pl or any(m in arrays_of_sym for m in members):
            continue
        # only pointers stored at offset 0 (or every 8 bytes): an array of pointers,
        # `struct node **head`, not a structure
        offs = {a[2] % 8 if a[1] is not None else a[2] for a in acc} | {c for _, _, c, _ in pl}
        if pl and offs == {0} and all(a[3] == "i64" for a in acc):
            for m in members:
                m.ptr_elem = pl[0][3]
            continue
        S = _layout(acc, pl, [k for m in members for k in steps.get(m, ())], find, root)
        if S is None:
            continue
        S._class = root
        for m in members:
            m.struct = S
        out.append(S)
    # pointer fields: target class -> its struct / element type
    for S in out:
        for f in S.fields.values():
            if f["ty"] == "ptr" and f.get("_dst") is not None:
                t = getattr(f["_dst"], "struct", None)
                f["target"] = t if t is not None else None
    return out


def _layout(acc, pl, steps, find, root):
    sizes = [BITS[a[3]] // 8 for a in acc] + [8 for _ in pl]
    amax = max(sizes) if sizes else 1
    # stride candidates
    ks = []
    for v, rest, c, ty, uns in acc:
        if rest is not None:
            k = stride_of(rest)
            if k > BITS[ty] // 8:
                ks.append(k)
    ks += [k for k in steps if k > amax]
    selfptr = any(find(d) is root for (_, _, _, d) in pl)
    K = None
    if ks:
        K = 0
        for k in ks:                 # an unrolled body (16*i) and its tail (8*j): period 8
            K = gcd(K, k)
        if K <= amax:
            K = max(set(ks), key=ks.count)
        if K <= amax:
            K = None                 # an unrolled walk over an array field, not a struct array
    # raw (offset, type) observations relative to the structure start
    obs = []
    arrays = []          # (base offset, elem type) for indexed accesses inside one structure
    for v, rest, c, ty, uns in acc:
        if K:
            if rest is not None and stride_of(rest) == 1 and BITS[ty] > 8:
                continue             # a byte offset kept in a variable: printed with a cast
            if rest is not None and stride_of(rest) % K != 0:
                return None          # indexed by something else: not a clean array of structs
            obs.append((c % K, ty, uns))
        else:
            if rest is None:
                obs.append((c, ty, uns))
            elif stride_of(rest) % (BITS[ty] // 8) == 0:
                arrays.append((c, ty))
            else:
                return None
    for b, rest, c, d in pl:
        if K:
            obs.append((c % K, "ptr", None))
        elif rest is None:
            obs.append((c, "ptr", None))
    if K:
        K = _period(obs, K, amax)
        obs = [(o % K, t, u) for o, t, u in obs]
    kinds = {t for _, t, _ in obs}
    offs = {o for o, _, _ in obs}
    if K:
        if K <= amax or (len(offs) < 2 and "ptr" not in kinds):
            return None
        if len(kinds) == 1 and next(iter(kinds)) in ("i8", "i16"):
            return None              # d[2*i], d[2*i+1]: an unrolled byte loop, not {char, char}
    else:
        sized = {t if t == "ptr" else BITS[t] for t in kinds} | ({"f"} if any(is_float(t) for t in kinds) else set())
        if any(c == 0 for c, _ in arrays):
            return None              # p[i] from the base: an array (fill: bytes, then 8-byte stores)
        tail_array = any(c not in offs and c > 0 for c, _ in arrays)   # b->len, b->data[i]
        structural = selfptr or (tail_array and offs) or len(sized) >= 2 or \
            (len(kinds) >= 2 and any(is_float(t) for t in kinds))
        if not structural:
            return None
    S = Struct()
    S.size = K
    S.stride = bool(K)
    # fields: per offset the narrowest observed type (a wide vector access over
    # two fields does not hide them), then drop overlaps
    per = {}
    for o, t, u in obs:
        per.setdefault(o, []).append((t, u))
    for o in sorted(per):
        ts = per[o]
        names = [x for x, _ in ts]
        if "ptr" in names:
            t = "ptr"
        else:
            t = min(names, key=lambda x: (BITS[x] // 8, -names.count(x)))
        us = [u for x, u in ts if x == t and u is not None]
        prev = [p for p in S.fields if p + fsize(S.fields[p]) > o]
        if prev:
            continue
        S.fields[o] = {"ty": t, "uns": bool(us) and all(us), "target": None, "count": None}
        if t == "ptr":
            for b, rest, c, d in pl:
                if (c % K if K else c) == o:
                    S.fields[o]["_dst"] = d
    for c, ty in sorted(arrays):
        if c in S.fields:
            continue
        if any(p <= c < p + fsize(f) for p, f in S.fields.items()):
            continue
        sz = BITS[ty] // 8
        # data[2*i] and data[2*i + 1]: the second is the same array
        lower = [o for o, f in S.fields.items() if f["count"] == 0 and f["ty"] == ty and o < c
                 and (c - o) % sz == 0 and not any(o < p < c for p in S.fields)]
        if lower:
            continue
        S.fields[c] = {"ty": ty, "uns": False, "target": None, "count": 0}
    # array lengths: up to the next field, the last one is flexible
    offs = sorted(S.fields)
    for i, o in enumerate(offs):
        f = S.fields[o]
        if f["count"] == 0:
            s = BITS[f["ty"]] // 8
            if i + 1 < len(offs):
                f["count"] = max(1, (offs[i + 1] - o) // s)
            elif K:
                f["count"] = max(1, (K - o) // s)
            else:
                S.flexible = True
    if len(S.fields) < 2 and not any(f["ty"] == "ptr" for f in S.fields.values()):
        return None                  # a single scalar: a plain pointer
    if K is None and not S.flexible:
        end = max(o + fsize(f) * (f["count"] or 1) for o, f in S.fields.items())
        al = max(min(fsize(f), 8) for f in S.fields.values())
        S.size = (end + al - 1) // al * al
    return S


def _period(obs, K, amax):
    """smallest P | K, P > amax, such that the layout repeats every P bytes"""
    best = K
    for P in range(K // 2, amax, -1):
        if K % P:
            continue
        lay = {}
        ok = True
        for o, t, u in obs:
            k = (o % P, t)
            lay.setdefault(o % P, t)
            if lay[o % P] != t:
                ok = False
                break
        if ok and len({o // P for o, _, _ in obs}) >= 2:
            best = P
    return best


# ------------------------------------------------------------ naming and printing
SIZES = {}           # struct name -> size in bytes (for pointer arithmetic)
BYNAME = {}          # struct name -> Struct


class Registry:
    """struct definitions of the whole file (identical layouts share a name)"""

    def __init__(self):
        self.by_key = {}
        self.order = []

    def register(self, structs):
        for S in structs:
            _fold_byte_runs(S)
        order = sorted(structs, key=lambda S: (-sum(f["count"] is not None for f in S.fields.values()),
                                               -len(S.fields)))
        for S in order:
            self._reg(S)
        for S in self.order:             # a flexible array read up to [k]: at least k+1
            for o, f in sorted(S.fields.items()):
                if f["count"] == 0 and f.get("_min") and S.flexible:
                    f["count"] = f["_min"]
                    S.flexible = False
                    end = max(p + span(g) for p, g in S.fields.items())
                    al = max(min(fsize(g), 8) for g in S.fields.values())
                    S.size = (end + al - 1) // al * al
                    SIZES[S.name] = S.size

    def _reg(self, S):
        if S.name:
            return S
        busy = self.__dict__.setdefault("_busy", set())
        busy.add(id(S))
        try:
            for f in S.fields.values():
                t = f["target"]
                # A -> B -> A (two list types pointing to each other): the one
                # being registered is named after its targets that are not on the stack
                if isinstance(t, Struct) and t is not S and not t.name and id(t) not in busy:
                    self._reg(t)
        finally:
            busy.discard(id(S))
        k = S.key()
        if k in self.by_key:
            other = self.by_key[k]
            S.name = other.name
            S._rep = other if other is not S else None
            return other
        # a layout that fits into a known one (list_len sees only `next` of
        # list_sum's node): the same type, extended with the new fields
        best, nbest = None, 0
        for T in self.order:
            n = _compatible(S, T)
            if n > nbest:
                best, nbest = T, n
        if best is not None:
            _merge(S, best)
            S.name = best.name
            S._rep = best
            SIZES[best.name] = best.size if not best.flexible else None
            return best
        S.name = "s%d" % (len(self.order) + 1)
        BYNAME[S.name] = S
        SIZES[S.name] = S.size if not S.flexible else None
        self.by_key[k] = S
        self.order.append(S)
        return S

    def definitions(self):
        out = []
        for S in self.order:
            out.append("struct %s;" % S.name)
        done = set()

        def emit(S):
            if id(S) in done:
                return
            done.add(id(S))
            for f in S.fields.values():
                if f["ty"] == "struct":
                    emit(rep(f["target"]))
            out.append(definition(S))
        for S in self.order:
            emit(S)
        return out


def rep(S):
    """the struct a merged layout was folded into"""
    seen = 0
    while isinstance(S, Struct) and getattr(S, "_rep", None) is not None and seen < 64:
        S = S._rep
        seen += 1
    return S


def _same_type(f, g, S, T):
    if f["ty"] != g["ty"] or f["count"] != g["count"]:
        return False
    if f["ty"] == "struct":
        return rep(f["target"]) is rep(g["target"])
    if f["ty"] == "ptr":
        a, b = rep(f["target"]), rep(g["target"])
        a = T if a is S else a           # self references match each other
        return a is b or (a is None or b is None)
    return True


def _in_array(T, o, f):
    """offset of T's array field holding a scalar field f at o, else None"""
    if f["count"] is not None or f["ty"] in ("ptr", "struct"):
        return None
    for p, h in T.fields.items():
        if h["count"] is not None and h["ty"] == f["ty"] and p <= o and (o - p) % fsize(h) == 0 and \
                (h["count"] == 0 or o < p + span(h)):
            return p
    return None


def _compatible(S, T):
    """> 0 if S's layout fits into T without conflict and they are linked by
    strong evidence: a common pointer field (list_len sees only `next`) or an
    access inside one of T's array fields (buf_last reads data[19]).  Plain
    scalar layouts that merely fit (pt vs outer) stay distinct types."""
    if (S.stride or T.stride) and S.size and T.size and S.size != T.size:
        return 0                     # an array element has an exact size
    common = strong = 0
    for o, f in S.fields.items():
        g = T.fields.get(o)
        if g is not None:
            if not _same_type(f, g, S, T):
                return 0
            common += 1
            strong += f["ty"] in ("ptr", "struct") or f["count"] is not None
            continue
        if _in_array(T, o, f) is not None:
            common += 1
            strong += 1
            continue
        end = o + span(f) if f["count"] != 0 else 1 << 40
        for p, h in T.fields.items():
            pend = p + span(h) if h["count"] != 0 else 1 << 40
            if p < end and o < pend:
                return 0             # overlaps a different field
        if T.stride and end > T.size:
            return 0
    return common if strong else 0


def _merge(S, T):
    for o, f in S.fields.items():
        a = _in_array(T, o, f) if o not in T.fields else None
        if a is not None:
            h = T.fields[a]
            k = (o - a) // fsize(h)
            if h["count"] == 0:          # flexible: at least up to the element read
                h["_min"] = max(h.get("_min", 0), k + 1)
            continue
        if o not in T.fields:
            g = dict(f)
            if g["ty"] == "ptr" and rep(g["target"]) is S:
                g["target"] = T
            T.fields[o] = g
        elif T.fields[o]["ty"] == "ptr" and T.fields[o]["target"] is None and f["target"] is not None:
            T.fields[o]["target"] = T if f["target"] is S else f["target"]
    if S.stride and not T.stride:
        T.size, T.stride = S.size, True
    elif not T.stride and not T.flexible:
        end = max(o + span(f) for o, f in T.fields.items())
        al = max(min(fsize(f), 8) for f in T.fields.values())
        T.size = max(T.size or 0, (end + al - 1) // al * al)
    T.flexible = T.flexible or S.flexible


def field_ctype(S, f):
    from .types import CTYPE
    if f["ty"] == "struct":
        return "struct %s" % rep(f["target"]).name
    if f["ty"] == "ptr":
        t = rep(f["target"])
        if isinstance(t, Struct):
            return "struct %s *" % t.name
        return "void *"
    k = ("u" if f["uns"] else "i") + f["ty"][1:] if f["ty"][0] == "i" else f["ty"]
    return CTYPE[k]


def definition(S):
    lines = ["struct %s" % S.name, "{"]
    pos = 0
    for o in sorted(S.fields):
        f = S.fields[o]
        if o > pos:
            lines.append("    char _p%x[%d];" % (pos, o - pos))
        ct = field_ctype(S, f)
        sep = "" if ct.endswith("*") else " "
        if f["ty"] == "struct":
            lines.append("    %s f%x;" % (ct, o))
            pos = o + fsize(f)
            continue
        if f["count"] is not None:
            if f["count"]:
                lines.append("    %s%sf%x[%d];" % (ct, sep, o, f["count"]))
                pos = o + fsize(f) * f["count"]
            else:
                lines.append("    %s%sf%x[];" % (ct, sep, o))
                pos = o
        else:
            lines.append("    %s%sf%x;" % (ct, sep, o))
            pos = o + fsize(f)
    if S.size and S.size > pos and not S.flexible:
        lines.append("    char _p%x[%d];" % (pos, S.size - pos))
    lines.append("};")
    return "\n".join(lines)


# ------------------------------------------------------------ byte offsets kept in variables
def rescale_offsets(nodes, F=None):
    """v = i << 3; ... *(int *)(p + v); *(int *)(p + v + 4)  ->  v = i; ... p + v*8 ...
    A variable whose every definition is a multiple of K and which is only ever
    added (unscaled) to an address holds a byte offset; keeping the element
    index instead is exact and lets p[v].f4 be printed (rec_find -O1)."""
    from .devec import _roots, _exprs_of, _map_root
    from .types import scale_down
    from .simplify import simplify
    from .ir import map_expr
    roots = _roots(nodes)
    defs, other = {}, set()
    offset_uses = {}

    def terms(e, out):
        if isinstance(e, Bin) and e.op == "+":
            terms(e.a, out)
            terms(e.b, out)
        else:
            out.append(e)

    def mark_address(a, ty):
        ts = []
        terms(a, ts)
        if not any(isinstance(t, Var) and getattr(t.var, "pointer", False) for t in ts) and \
                not any(type(t).__name__ == "Sym" for t in ts):
            return {}
        size = BITS.get(ty, 64) // 8
        return {id(t): size for t in ts if isinstance(t, Var) and not getattr(t.var, "pointer", False)}

    feeds = {}                       # var -> vars whose definition it is added into
    narrow = set()                   # offsets of byte accesses: already element indices
    for h in roots:
        dst = h.dst.var if isinstance(h, Assign) and isinstance(h.dst, Var) else None
        if dst is not None:
            defs.setdefault(dst, []).append(h)
        src_ids = {id(x) for x in h.src.walk()} if dst is not None else set()
        ok_ids = {}
        exprs = list(_exprs_of(h))
        if isinstance(h, Store):
            ok_ids.update(mark_address(h.addr, h.ty))
        for e in exprs:
            for x in e.walk():
                if isinstance(x, Load):
                    ok_ids.update(mark_address(x.addr, x.ty))
        for e in exprs:
            for x in e.walk():
                if isinstance(x, Var):
                    if id(x) in ok_ids:
                        offset_uses[x.var] = offset_uses.get(x.var, 0) + 1
                        if ok_ids[id(x)] == 1:
                            narrow.add(x.var)
                    elif id(x) in src_ids:
                        # v = v + 8, v11 = v73: offsets flowing into another offset
                        # (checked below: the definition must scale down)
                        feeds.setdefault(x.var, set()).add(dst)
                    else:
                        other.add(x.var)
    cand = {v for v in set(offset_uses) | set(feeds)
            if v in defs and v not in other and v not in narrow and v.ty == "i64"
            and not getattr(v, "pointer", False) and not v.is_param and not v.is_input}

    def st(e, K):
        """stride of e, a variable of the plan counting as K[v] (0: any)"""
        if isinstance(e, Var) and e.var in K:
            return K[e.var]
        if isinstance(e, Bin) and e.op in ("+", "-"):
            return gcd(st(e.a, K), st(e.b, K))
        if isinstance(e, Bin) and e.op == "*" and isinstance(e.b, Const):
            return abs(sext(e.b.value, 64)) * st(e.a, K)
        if isinstance(e, Bin) and e.op == "<<" and isinstance(e.b, Const) and e.b.value < 32:
            return (1 << e.b.value) * st(e.a, K)
        if isinstance(e, Un) and e.op == "-":
            return st(e.a, K)
        return stride_of(e)

    def scaled(e, K):
        return map_expr(e, lambda x: Bin("*", x, Const(K[x.var], "i64"), "i64")
                        if isinstance(x, Var) and x.var in K and x.ty == "i64" else x)

    plan = {}
    while True:
        drop = True
        while drop:
            drop = {v for v in cand if any(d not in cand for d in feeds.get(v, ()))}
            cand -= drop
        # per variable the largest K dividing every definition (fixpoint: K only shrinks)
        K = {v: 0 for v in cand}
        changed = True
        while changed:
            changed = False
            for v in cand:
                k = K[v]
                for d in defs[v]:
                    k = gcd(k, st(d.src, K))
                if k != K[v]:
                    K[v] = k
                    changed = True
        bad = {v for v in cand if K[v] <= 1 or
               any(scale_down(scaled(d.src, K), K[v]) is None for d in defs[v])}
        # a group of offsets feeding each other is worth it only if one is an address
        comp = {v: {v} for v in cand}
        for v in cand:
            for d in feeds.get(v, ()):
                if comp[v] is not comp[d]:
                    merged = comp[v] | comp[d]
                    for x in merged:
                        comp[x] = merged
        for g in {id(g): g for g in comp.values()}.values():
            if not any(offset_uses.get(v) for v in g):
                bad |= g
        if not bad:
            plan = K
            break
        cand -= bad
    if not plan:
        return False
    news = {v: [simplify(scale_down(scaled(d.src, plan), plan[v])) for d in defs[v]] for v in plan}
    for v in plan:
        for d, n in zip(defs[v], news[v]):
            d.src = n

    def fix(e):
        def f(x):
            if isinstance(x, Var) and x.var in plan and x.ty == "i64":
                return Bin("*", x, Const(plan[x.var], "i64"), "i64")
            return x
        return map_expr(e, f)
    for h in roots:
        if isinstance(h, Assign) and isinstance(h.dst, Var) and h.dst.var in plan:
            if h.guard is not None:
                h.guard = fix(h.guard)
            continue
        _map_root(h, fix)
    return True
