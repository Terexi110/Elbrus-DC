"""Types across functions of one file.

After every function has its AST and its own type facts (structs.infer,
types.pointees), this pass connects them:

* call arguments and callee parameters: a pointer passed to a parameter of a
  known structure type gets that type (and the other way round);
* `p + c` passed where a structure is expected: the structure of p embeds
  that structure at offset c (nested structure);
* `&local` passed where a structure is expected: the stack object is that
  structure;
* global data symbols accessed at several offsets with different types (or
  indexed with a stride) are structures / arrays of structures.

Then all structures are registered (names, merging) for printing.
"""
from .ir import Var, Sym, Bin, Const, Call, Load, Store, Assign, Cast, BITS, sext
from .structs import Struct, const_part, lin, rep, fsize


def _calls(ast):
    from .devec import _roots, _exprs_of
    for h in _roots(ast):
        for e in _exprs_of(h):
            for x in e.walk():
                if isinstance(x, Call) and isinstance(x.target, Sym) and x.target.kind == "func":
                    yield x


def _params(F):
    out = {}
    for v in F.vars:
        if v.is_param and v.param_index is not None:
            out[v.param_index] = v
    return out


def _split_ptr_arg(a):
    """arg -> (pointer VarInfo, constant offset) for p / p + c, or (Sym, off) for &sym + c"""
    while isinstance(a, Cast) and BITS.get(a.ty, 64) == 64:
        a = a.a
    if isinstance(a, Var) and getattr(a.var, "pointer", False):
        return a.var, 0
    if isinstance(a, Sym) and a.kind in ("local", "data"):
        return a, a.off
    if isinstance(a, Bin) and a.op == "+":
        rest, c = const_part(a)
        if rest is not None and c != 0:          # p + i (no constant): not an offset
            r = _split_ptr_arg(rest)
            if r is not None:
                return r[0], r[1] + c
    return None


def link_types(items, ctx):
    """items: [(name, F, (ast, found_structs))]"""
    funcs = {name: (F, p) for name, F, p in items}
    params = {name: _params(F) for name, (F, _) in funcs.items()}
    all_found = []
    for name, F, (ast, found) in items:
        all_found.extend(found)
    # 1. arguments <-> parameters (a few rounds: types flow both ways)
    for _ in range(3):
        changed = False
        for name, F, (ast, found) in items:
            for call in _calls(ast):
                ps = params.get(call.target.name)
                if not ps:
                    continue
                for i, a in enumerate(call.args):
                    pv = ps.get(i)
                    if pv is None:
                        continue
                    r = _split_ptr_arg(a)
                    if r is None:
                        continue
                    base, off = r
                    S = rep(getattr(pv, "struct", None))
                    if isinstance(base, Sym):
                        if S is not None and base.kind == "local":
                            changed |= _stack_struct(F, base, off, S)
                        continue
                    T = rep(getattr(base, "struct", None))
                    if off == 0:
                        if S is not None and T is None:
                            base.struct = S
                            changed = True
                        elif T is not None and S is None and getattr(pv, "pointer", False):
                            pv.struct = T
                            changed = True
                    elif S is not None and T is not None:
                        changed |= _embed(T, off, S)
                    elif T is not None and getattr(pv, "pointer", False) and \
                            isinstance(getattr(pv, "pointee", None), str):
                        changed |= _embed_array(T, off, pv.pointee)
        if not changed:
            break
    # 2. global structures
    _global_structs(items, ctx, all_found)
    # 3. names and merging
    ctx.structs.register(all_found)


def _embed(T, off, S):
    """T contains a whole S at offset off (nested structure)"""
    from .structs import rep
    f = T.fields.get(off)
    if f is not None and f["ty"] == "struct":
        return False

    def contains(A, B, depth=0):
        A = rep(A)
        if A is rep(B):
            return True
        return depth < 16 and any(g["ty"] == "struct" and contains(g["target"], B, depth + 1)
                                  for g in A.fields.values())
    if contains(S, T):
        return False                     # a structure cannot contain itself
    size = S.size or 0
    if not size:
        return False
    # fields of T inside [off, off+size) belong to S now
    inside = [o for o in T.fields if off <= o < off + size]
    for o in inside:
        g = T.fields[o]
        if g["ty"] == "struct":
            return False
        sf = S.fields.get(o - off)
        if sf is None and g["ty"] != "ptr":
            S.fields[o - off] = dict(g)
        del T.fields[o]
    T.fields[off] = {"ty": "struct", "uns": False, "target": S, "count": None}
    if T.size is not None and not T.stride and off + size > T.size:
        T.size = (off + size + 7) // 8 * 8
    return True


def _embed_array(T, off, el):
    """&p->field passed where an `el *` is expected: an array of el at off"""
    from .types import ir_ty
    ty = ir_ty(el)
    if ty not in ("i8", "i16", "i32", "i64", "f32", "f64"):
        return False
    sz = BITS[ty] // 8
    if off in T.fields:
        return False
    if any(o < off < o + fsize(f) for o, f in T.fields.items()):
        return False
    nxt = min((o for o in T.fields if o > off), default=None)
    if nxt is None:
        if not T.size:
            return False
        nxt = T.size
    if (nxt - off) % sz:
        return False
    T.fields[off] = {"ty": ty, "uns": el[0] == "u", "target": None, "count": (nxt - off) // sz}
    return True


def _stack_struct(F, sym, off, S):
    """the stack object at sym+off is an S: one local of that type"""
    locs = getattr(F, "locals", [])
    base = next((L for L in locs if L["name"] == sym.name), None)
    if base is None or not S.size:
        return False
    start = base["off"] + off
    if base.get("struct") is S and base["off"] == start:
        return False
    # every local inside [start, start + size) becomes part of the object
    merged = [L for L in locs if start <= L["off"] < start + S.size]
    name = "local_%x" % (-start)
    obj = {"name": name, "off": start, "ty": "i8", "count": S.size, "struct": S}
    F.locals = [L for L in locs if L not in merged] + [obj]
    F.local_alias = getattr(F, "local_alias", {})
    for L in merged:
        F.local_alias[L["name"]] = (name, L["off"] - start)
    return True


def _global_structs(items, ctx, all_found):
    """data symbols accessed at offsets of different types / with a stride"""
    from .devec import _roots, _exprs_of
    from .structs import _layout
    acc = {}
    for name, F, (ast, found) in items:
        def access(addr, ty, uns):
            if ty not in ("i8", "i16", "i32", "i64", "f32", "f64"):
                return
            a = lin(addr)
            rest, c = const_part(a)
            sym = None
            if isinstance(rest, Sym):
                sym, rest = rest, None
            elif isinstance(rest, Bin) and rest.op == "+":
                for x, y in ((rest.a, rest.b), (rest.b, rest.a)):
                    if isinstance(x, Sym):
                        sym, rest = x, y
                        break
            if sym is None or sym.kind != "data":
                return
            acc.setdefault(sym.name, []).append((None, rest, c + sym.off, ty, uns))
        for h in _roots(ast):
            if isinstance(h, Store):
                access(h.addr, h.ty, None)
            for e in _exprs_of(h):
                for x in e.walk():
                    if isinstance(x, Load):
                        access(x.addr, x.ty, None)
    for sym, a in acc.items():
        # how each data symbol is accessed: element types, constant offsets,
        # indexed accesses (decides `g` / `g[k]` when printing)
        ctx.global_use[sym] = ({x[3] for x in a}, {x[2] for x in a}, any(x[1] is not None for x in a))
        if ctx.string_at(Sym(sym, 0, "data")) is not None:
            continue
        S = _layout(a, [], [], lambda v: v, None)
        if S is None:
            continue
        size = ctx.symbol_size(sym)
        count = None
        if S.stride and size and size % S.size == 0:
            count = size // S.size
        elif not S.stride and size:
            S.size = max(S.size or 0, size)
        S._class = None
        ctx.global_structs[sym] = (S, count)
        all_found.append(S)
