"""Lifting of e2k wide instructions (bundles) to IR effects.

Semantics model:
  * every operation of a wide instruction reads the register state that
    existed *before* the instruction and writes after it (parallel issue);
  * PLU operations (pass/andp/landp) are evaluated in textual order,
    local predicates @pN live only inside the wide instruction;
  * control transfer (ct/call) happens after the operations of its bundle.
"""
import re

from .ir import (Const, Reg, Bin, Un, Cast, Load, Select, Intr, Sym, Assign, Store,
                 ExprStmt, is_float, BITS)

RE_REG = re.compile(r"^%(d|x|q|)(r|g)(\d+)$")
RE_BREG = re.compile(r"^%(d|x|q|)b\[(\d+)\]$")
RE_LIT = re.compile(r"^_(f16|f32|f64|lit16|lit32|lit64|lts\d)(s?)\s+(0x[0-9a-f]+)$")
RE_LITC = re.compile(r"^_(f16|f32|f64)(s?),_lts\d\s+(0x[0-9a-f]+)$")

VIEW_TY = {"": "i32", "d": "i64", "x": "i64", "q": "i128"}

INT_BIN = {
    "add": "+", "sub": "-", "mul": "*", "and": "&", "or": "|", "xor": "^",
    "shl": "<<", "shr": ">>u", "sar": ">>s", "scl": "rotl", "scr": "rotr",
    "sdiv": "/s", "udiv": "/u", "andn": "&~", "orn": "|~", "xorn": "^~",
}
FP_BIN = {"fadd": "+", "fsub": "-", "fmul": "*", "fdiv": "/", "fmin": "fmin", "fmax": "fmax"}

INT_CMP = {"e": "==", "l": "<s", "le": "<=s", "b": "<u", "be": "<=u"}
FP_CMP = {"eq": "==f", "lt": "<f", "le": "<=f", "neq": "!=f", "nlt": "!<f", "nle": "!<=f",
          "uod": "uod", "od": "od"}

LOADS = {"ldb": "i8", "ldh": "i16", "ldw": "i32", "ldd": "i64", "ldq": "i128"}
STORES = {"stb": "i8", "sth": "i16", "stw": "i32", "std": "i64", "stq": "i128"}

CONV = {  # mnemonic: (src_ty, dst_ty, kind)
    "istofs": ("i32", "f32", "i2f"), "istofd": ("i32", "f64", "i2f"),
    "idtofs": ("i64", "f32", "i2f"), "idtofd": ("i64", "f64", "i2f"),
    "fstois": ("f32", "i32", "f2i"), "fstoistr": ("f32", "i32", "f2i"),
    "fdtois": ("f64", "i32", "f2i"), "fdtoistr": ("f64", "i32", "f2i"),
    "fstoid": ("f32", "i64", "f2i"), "fstoidtr": ("f32", "i64", "f2i"),
    "fdtoid": ("f64", "i64", "f2i"), "fdtoidtr": ("f64", "i64", "f2i"),
    "fstofd": ("f32", "f64", "f2f"), "fdtofs": ("f64", "f32", "f2f"),
}

IGNORED = {"nop", "ipd", "setwd", "setbn", "setbp", "vfrpsz", "loop_mode", "abn", "abp",
           "alc", "bap", "eap", "wait", "setei", "pref", "flushr", "flushc", "incr",
           "settr", "setsft", "fapb"}


def parse_imm(s):
    s = s.strip()
    if re.match(r"^0x[0-9a-f]+$", s):
        return int(s, 16), None
    m = RE_LIT.match(s) or RE_LITC.match(s)
    if m:
        kind, sgn, val = m.group(1), m.group(2), int(m.group(3), 16)
        bits = 16 if "16" in kind else 32 if "32" in kind else 64 if "64" in kind else 32
        if sgn and val >> (bits - 1):
            val -= 1 << bits
        return val, kind
    return None, None


class BundleLift:
    """Result of lifting one wide instruction."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.stmts = []        # IR statements in executable order
        self.ct = None         # (ctpr, cond-expr or None)
        self.ibranch = None    # (target_addr, cond-expr or None)
        self.call = None       # (ctpr, wbs, cond)
        self.ctpr_set = {}     # ctpr -> descriptor
        self.setbn = None      # (rbs, rsz, rcur)
        self.getsp = []
        self.unknown = []
        self.loop = False      # bundle uses pipelined-loop machinery


class Lifter:
    def __init__(self, func, resolver):
        self.func = func
        self.res = resolver    # symbol resolver (see decompiler.Resolver)

    # ------------------------------------------------------------ operands
    def breg(self, idx, bn):
        rbs, rsz, rcur = bn if bn else (0, 0, 0)
        size = 2 * (rsz + 1)
        return "r%d" % (2 * rbs + (idx + 2 * rcur) % max(size, 1))

    def reg_of(self, s, bn):
        """'%dr5' -> ('r5','i64'), '%b[2]' -> ('r10','i32')..."""
        m = RE_REG.match(s)
        if m:
            return ("%s%s" % (m.group(2), m.group(3))), VIEW_TY[m.group(1)]
        m = RE_BREG.match(s)
        if m:
            return self.breg(int(m.group(2)), bn), VIEW_TY[m.group(1)]
        m = re.match(r"^%pred(\d+)$", s)
        if m:
            return "pred" + m.group(1), "b"
        return None, None

    def read(self, s, ty, bn, lits):
        """Operand text -> Expr of type ty."""
        s = s.strip()
        name, view = self.reg_of(s, bn)
        if name:
            if view == "b":
                return Reg(name, "b")
            if is_float(ty):
                return Reg(name, ty)
            e = Reg(name, view)
            if view != ty and ty in ("i32", "i64") and view in ("i32", "i64"):
                return Cast(e, ty, signed=True)
            return e
        val, kind = parse_imm(s)
        if val is not None:
            if kind in ("f32", "f64") and lits:
                rel = lits.pop(0)
                return self.res.reloc_expr(rel, val)
            if is_float(ty):
                return self._fconst(val, ty)
            return Const(val, ty if ty in ("i8", "i16", "i32", "i64") else "i64")
        if s.startswith("%"):
            return Intr("reg_" + s[1:].replace("[", "_").replace("]", ""), [], ty, pure=False)
        return Intr("unparsed", [], ty, pure=False)

    def _fconst(self, val, ty):
        # literal used as float operand: reinterpret bits
        import struct
        from .ir import FConst
        if ty == "f64":
            return FConst(struct.unpack("<d", struct.pack("<Q", val & (2 ** 64 - 1)))[0], "f64")
        return FConst(struct.unpack("<f", struct.pack("<I", val & (2 ** 32 - 1)))[0], "f32")

    def cond(self, s, bn):
        if s is None:
            return None
        s = s.strip()
        if "||" in s:
            a, b = s.split("||", 1)
            return Bin("||", self.cond(a, bn), self.cond(b, bn), "b")
        if "&&" in s:
            a, b = s.split("&&", 1)
            return Bin("&&", self.cond(a, bn), self.cond(b, bn), "b")
        neg = s.startswith("~")
        if neg:
            s = s[1:].strip()
        m = re.match(r"^%pred(\d+)$", s)
        if m:
            e = Reg("pred" + m.group(1), "b")
        elif s in ("%NOT_LOOP_END", "%LOOP_END", "%MLOCK"):
            e = Intr(s[1:].lower(), [], "b", pure=False)
        else:
            m = re.match(r"^%pcnt(\d+)$", s)
            e = Intr("pcnt", [Const(int(m.group(1)), "i32")], "b", pure=False) if m else \
                Intr("cond", [], "b", pure=False)
        return Un("!", e, "b") if neg else e

    def dst(self, s, bn):
        name, view = self.reg_of(s, bn)
        if name:
            return name, view
        return None, None

    # ------------------------------------------------------------ bundles
    def lift(self, bundle, bn):
        L = BundleLift(bundle)
        lits = [r for r in bundle.relocs if "DISP" not in r[1]]
        lits.sort(key=lambda r: r[0])
        effects = []           # ('set', name, ty, expr, guard) | ('store', addr, val, ty, guard) | ('eval', e, guard)
        lp = {}                # local predicates @pN -> expr
        plu_writes = []
        readers = []
        # PLU executes in stages: pass %pred->@p, then landp/andp (by result
        # index, results may cascade), then pass @p->%pred.  objdump prints
        # them in encoding order, so reorder here.
        def plu_stage(op):
            if op.mnem == "pass":
                return 0 if not op.args[0].strip().startswith("@p") else 2
            if op.mnem in ("andp", "landp"):
                return 1
            return -1

        def plu_key(op):
            st = plu_stage(op)
            idx = int(re.sub(r"\D", "", op.args[-1])) if st == 1 else 0
            return (st, idx)
        normal = [o for o in bundle.ops if plu_stage(o) < 0]
        plu = sorted([o for o in bundle.ops if plu_stage(o) >= 0], key=plu_key)
        for op in normal + plu:
            m = op.mnem
            a = [x for x in op.args if not x.startswith("mas")]
            guard = self.cond(op.guard, bn) if m not in ("ct", "ibranch", "call") else None
            try:
                self._lift_op(L, op, m, a, guard, bn, lits, effects, lp, plu_writes)
            except Exception as ex:  # robust: never die on one op
                L.unknown.append(op.text.strip())
                effects.append(("eval", Intr("asm", [], "v", pure=False), None, op.text.strip()))
        effects.extend(plu_writes)
        # ---- serialize parallel writes
        written = {}
        for ef in effects:
            if ef[0] == "set":
                written.setdefault(ef[1], 0)
                written[ef[1]] += 1
        read_names = []
        for ef in effects:
            exprs = []
            if ef[0] == "set":
                exprs = [ef[3]] + ([ef[4]] if ef[4] is not None else [])
            elif ef[0] == "store":
                exprs = [ef[1], ef[2]] + ([ef[4]] if ef[4] is not None else [])
            elif ef[0] == "eval":
                exprs = [ef[1]] + ([ef[2]] if ef[2] is not None else [])
            names = set()
            for e in exprs:
                for x in e.walk():
                    if isinstance(x, Reg):
                        names.add(x.name)
            read_names.append(names)
        ctl_reads = set()
        for c in (L.ct[1] if L.ct else None, L.ibranch[1] if L.ibranch else None,
                  L.call[2] if L.call else None):
            if c is not None:
                ctl_reads |= {x.name for x in c.walk() if isinstance(x, Reg)}
        deferred = []
        tmpn = [0]
        for i, ef in enumerate(effects):
            if ef[0] == "set":
                _, name, ty, expr, guard = ef
                conflict = name in ctl_reads or any(name in rn for j, rn in enumerate(read_names) if j != i)
                if conflict:
                    tmpn[0] += 1
                    t = "t%x_%d" % (bundle.addr, tmpn[0])
                    pure = not any(isinstance(x, Intr) and not x.pure for x in expr.walk())
                    L.stmts.append(Assign(Reg(t, ty), expr, None if pure else guard, addr=bundle.addr))
                    deferred.append(Assign(Reg(name, ty), Reg(t, ty), guard, addr=bundle.addr))
                else:
                    L.stmts.append(Assign(Reg(name, ty), expr, guard, addr=bundle.addr))
            elif ef[0] == "store":
                _, addr, val, ty, guard = ef
                L.stmts.append(Store(addr, val, ty, guard, at=bundle.addr))
            else:
                st = ExprStmt(ef[1], ef[2], addr=bundle.addr)
                if len(ef) > 3:
                    st.asm = ef[3]
                L.stmts.append(st)
        L.stmts.extend(deferred)
        return L

    def _lift_op(self, L, op, m, a, guard, bn, lits, effects, lp, plu_writes):
        R = lambda s, ty: self.read(s, ty, bn, lits)

        def setr(dst, expr, ty):
            name, view = self.dst(dst, bn)
            if name is None:
                if dst.strip() in ("%empty", "%tst", "%tc", "%tcd"):
                    return
                effects.append(("eval", Intr("write_" + dst.strip("%"), [expr], "v", pure=False), guard))
                return
            if view == "b":
                ty = "b"
            elif not is_float(ty):
                if view != ty and BITS.get(view, 64) < BITS.get(ty, 64):
                    expr = Cast(expr, view, signed=False)
                    ty = view
                elif view != ty and ty in ("i8", "i16", "i32") and view in ("i32", "i64"):
                    expr = Cast(expr, view, signed=False)
                    ty = view
            effects.append(("set", name, ty, expr, guard))

        if m in IGNORED:
            if m == "setbn":
                d = dict(re.findall(r"(\w+) = (0x[0-9a-f]+)", " ".join(op.args)))
                L.setbn = (int(d.get("rbs", "0"), 16), int(d.get("rsz", "0"), 16), int(d.get("rcur", "0"), 16))
            if m in ("loop_mode", "abn", "alc", "abp"):
                L.loop = True
            return
        # ---------------- control
        if m == "disp":
            L.ctpr_set[a[0]] = ("disp", self.res.disp_target(L.bundle, a[1]))
            return
        if m == "sdisp":
            L.ctpr_set[a[0]] = ("syscall", a[1] if len(a) > 1 else "")
            return
        if m == "return":
            L.ctpr_set[a[0]] = ("ret",)
            return
        if m in ("ldisp", "gdisp"):
            L.ctpr_set[a[0]] = ("unknown",)
            return
        if m == "ct":
            L.ct = (a[0], self.cond(op.guard, bn))
            return
        if m == "ibranch":
            L.ibranch = (int(a[0], 16), self.cond(op.guard, bn))
            return
        if m == "call":
            wbs = int(re.search(r"wbs = (0x[0-9a-f]+)", " ".join(a)).group(1), 16)
            L.call = (a[0], wbs, self.cond(op.guard, bn))
            return
        if m == "movtd" and a[-1].startswith("%ctpr"):
            L.ctpr_set[a[-1]] = ("reg", R(a[0], "i64"))
            return
        # ---------------- PLU
        if m == "pass":
            src = a[0].strip()
            if src.startswith("@p"):
                e = lp.get(src, Intr("lp", [], "b"))
            else:
                e = self.cond(src, bn)
            dst = a[1].strip()
            if dst.startswith("@p"):
                lp[dst] = e
            else:
                name, _ = self.dst(dst, bn)
                plu_writes.append(("set", name, "b", e, None))
            return
        if m in ("andp", "landp"):
            def lpv(s):
                s = s.strip()
                neg = s.startswith("~")
                s = s.lstrip("~")
                e = lp.get(s, Intr("lp", [], "b"))
                return Un("!", e, "b") if neg else e
            lp[a[2].strip()] = Bin("&&", lpv(a[0]), lpv(a[1]), "b")
            return
        # ---------------- ALU
        base, sfx = m[:-1], m[-1:]
        ity = {"s": "i32", "d": "i64"}.get(sfx)
        if m == "getsp":
            v, _ = parse_imm(a[0])
            if v is None:
                v = 0
                src = R(a[0], "i32")
                effects.append(("set", self.dst(a[1], bn)[0], "i64",
                                Bin("+", Reg("sp", "i64"), Cast(src, "i64"), "i64"), guard))
                return
            L.getsp.append(v)
            setr(a[1], Bin("+", Reg("sp", "i64"), Const(v, "i64"), "i64"), "i64")
            return
        if m in ("addd", "ord", "xord") and len(a) == 3 and \
                (parse_imm(a[0])[0] == 0 or parse_imm(a[1])[0] == 0) and guard is None:
            # 64-bit register move: type-agnostic copy
            other = a[1] if parse_imm(a[0])[0] == 0 else a[0]
            name, view = self.reg_of(other, bn)
            dn, dv = self.dst(a[2], bn)
            if name and view == "i64" and dn and dv == "i64":
                effects.append(("set", dn, "any", Reg(name, "any"), guard))
                return
        if ity and base in INT_BIN and len(a) == 3:
            ty = ity
            x, y = R(a[0], ty), R(a[1], ty)
            op_ = INT_BIN[base]
            if op_ in ("&~", "|~", "^~"):
                e = Bin(op_[0], x, Un("~", y, ty), ty)
            else:
                e = Bin(op_, x, y, ty)
            setr(a[2], e, ty)
            return
        # fused 3-input ops: e.g. add_subd, sar_rsubd
        mm = re.match(r"^([a-z]+)_(r?[a-z]+)([sd])$", m)
        if mm and mm.group(1) in INT_BIN and mm.group(2).lstrip("r") in INT_BIN and len(a) == 4:
            ty = {"s": "i32", "d": "i64"}[mm.group(3)]
            x, y, z = R(a[0], ty), R(a[1], ty), R(a[2], ty)
            r1 = Bin(INT_BIN[mm.group(1)], x, y, ty)
            o2 = mm.group(2)
            if o2.startswith("r") and o2 != "rsub" and o2[1:] not in INT_BIN:
                o2 = o2
            if o2 == "rsub":
                e = Bin("-", z, r1, ty)
            else:
                e = Bin(INT_BIN[o2.lstrip("r")], r1, z, ty)
            setr(a[3], e, ty)
            return
        if m in ("umulx", "smulx"):
            sg = m[0] == "s"
            e = Bin("*", Cast(R(a[0], "i32"), "i64", sg), Cast(R(a[1], "i32"), "i64", sg), "i64")
            setr(a[2], e, "i64")
            return
        if m in ("umulhd", "smulhd"):
            setr(a[2], Intr(m[:5], [R(a[0], "i64"), R(a[1], "i64")], "i64"), "i64")
            return
        if m in ("udivx", "sdivx", "umodx", "smodx"):
            sg = m[0] == "s"
            o = ("/" if "div" in m else "%") + ("s" if sg else "u")
            e = Cast(Bin(o, R(a[0], "i64"), Cast(R(a[1], "i32"), "i64", sg), "i64"), "i32", sg)
            setr(a[2], e, "i32")
            return
        if m == "sxt":
            v, _ = parse_imm(a[0])
            src = R(a[1], "i32")
            if v is None:
                setr(a[2], Intr("sxt", [R(a[0], "i32"), src], "i64"), "i64")
                return
            v &= 7
            width = {0: 8, 4: 8, 1: 16, 5: 16}.get(v, 32)
            signed = not (v & 4)
            e = Cast(Cast(src, "i%d" % width, signed=False), "i64", signed=signed) if width < 32 \
                else Cast(src, "i64", signed=signed)
            setr(a[2], e, "i64")
            return
        if m in ("getfs", "getfd"):
            ty = "i32" if m == "getfs" else "i64"
            bits = 32 if ty == "i32" else 64
            v, _ = parse_imm(a[1])
            src = R(a[0], ty)
            if v is None:
                setr(a[2], Intr(m, [src, R(a[1], ty)], ty), ty)
                return
            sh = v & (31 if bits == 32 else 63)
            size = (v >> 6) & (31 if bits == 32 else 63)
            sgn = bool(v & (1 << 12))
            if size == 0:
                e = Const(0, ty)
            else:
                x = Bin(">>u", src, Const(sh, ty), ty) if sh else src
                if size in (8, 16, 32) and size < bits:
                    e = Cast(Cast(x, "i%d" % size, signed=False), ty, signed=sgn)
                elif sgn:
                    e = Bin(">>s", Bin("<<", x, Const(bits - size, ty), ty), Const(bits - size, ty), ty)
                else:
                    e = Bin("&", x, Const((1 << size) - 1, ty), ty)
            setr(a[2], e, ty)
            return
        if m in ("merges", "merged"):
            ty = "i32" if m == "merges" else "i64"
            c = self.cond(a[3], bn) if len(a) > 3 else Intr("mrgc", [], "b")
            setr(a[2], Select(c, R(a[1], ty), R(a[0], ty), ty), ty)
            return
        if m.startswith("cmp") and m.endswith(("sb", "db")):
            ty = "i32" if m.endswith("sb") else "i64"
            core = m[3:-2]
            x, y = R(a[0], ty), R(a[1], ty)
            if core.startswith("and"):
                k = core[3:]
                t = Bin("&", x, y, ty)
                z = Const(0, ty)
                e = {"e": Bin("==", t, z, "b"), "s": Bin("<s", t, z, "b"),
                     "le": Bin("<=s", t, z, "b"), "p": Intr("parity", [t], "b")}.get(k)
            elif core in INT_CMP:
                e = Bin(INT_CMP[core], x, y, "b")
            elif core == "s":
                e = Bin("<s", Bin("-", x, y, ty), Const(0, ty), "b")
            elif core == "o":
                e = Intr("sub_overflow", [x, y], "b")
            elif core == "p":
                e = Intr("parity", [Bin("-", x, y, ty)], "b")
            else:
                e = None
            if e is None:
                raise ValueError(m)
            setr(a[2], e, "b")
            return
        mm = re.match(r"^fcmp([a-z]+?)(s|d)b$", m)
        if mm and mm.group(1) in FP_CMP:
            ty = "f32" if mm.group(2) == "s" else "f64"
            x, y = R(a[0], ty), R(a[1], ty)
            o = FP_CMP[mm.group(1)]
            if o.startswith("!"):
                e = Un("!", Bin(o[1:], x, y, "b"), "b")
            elif o in ("uod", "od"):
                e = Intr("unordered", [x, y], "b")
                if o == "od":
                    e = Un("!", e, "b")
            else:
                e = Bin(o, x, y, "b")
            setr(a[2], e, "b")
            return
        if m[:-1] in FP_BIN and m[-1] in "sd" and len(a) == 3:
            ty = "f32" if m[-1] == "s" else "f64"
            o = FP_BIN[m[:-1]]
            x, y = R(a[0], ty), R(a[1], ty)
            e = Bin(o + "f" if o in "+-*/" else o, x, y, ty) if o in "+-*/" else Intr(o, [x, y], ty)
            setr(a[2], e, ty)
            return
        if m in CONV:
            st, dt, kind = CONV[m]
            setr(a[1], Cast(R(a[0], st), dt, signed=True), dt)
            return
        if m in LOADS:
            ty = LOADS[m]
            addr = Bin("+", R(a[0], "i64"), R(a[1], "i64"), "i64")
            e = Load(addr, ty)
            name, view = self.dst(a[2], bn)
            if name is None:
                return
            if ty in ("i8", "i16"):
                e = Cast(e, view if view in ("i32", "i64") else "i32", signed=False)
                effects.append(("set", name, view if view in ("i32", "i64") else "i32", e, guard))
            else:
                effects.append(("set", name, ty, e, guard))
            return
        if m in STORES:
            ty = STORES[m]
            addr = Bin("+", R(a[0], "i64"), R(a[1], "i64"), "i64")
            vty = "i64" if ty in ("i64", "i128") else "i32"
            val = R(a[2], vty)
            if ty in ("i8", "i16"):
                val = Cast(val, ty, signed=False)
            effects.append(("store", addr, val, ty, guard))
            return
        if m in ("movtd", "movts", "movtq", "movtcd", "movtcs"):
            ty = "i32" if m.endswith("s") else "i64"
            setr(a[-1], R(a[0], ty), ty)
            return
        if m in ("puttags", "puttagd"):
            ty = "i32" if m.endswith("s") else "i64"
            setr(a[2], R(a[0], ty), ty)
            return
        if m in ("gettags", "gettagd"):
            setr(a[1], Const(0, "i32"), "i32")
            return
        if m in ("rrs", "rrd"):
            ty = "i32" if m == "rrs" else "i64"
            setr(a[1], Intr("rr", [Intr("sr_" + a[0].strip("%"), [], ty)], ty, pure=False), ty)
            return
        if m in ("rws", "rwd"):
            ty = "i32" if m == "rws" else "i64"
            effects.append(("eval", Intr("rw_" + a[1].strip("%"), [R(a[0], ty)], "v", pure=False), guard))
            return
        if m in ("lzcnts", "lzcntd", "popcnts", "popcntd", "bitrevs", "bitrevd"):
            ty = "i32" if m.endswith("s") else "i64"
            setr(a[-1], Intr(m[:-1], [R(a[0], ty)], ty), ty)
            return
        if m in ("fabss", "fabsd", "fsqrts", "fsqrtd"):
            ty = "f32" if m.endswith("s") else "f64"
            setr(a[-1], Intr(m[:-1], [R(a[-2], ty)], ty), ty)
            return
        if m.startswith(("staa", "aau", "ldaa", "aaur", "mova")):
            L.loop = True
            effects.append(("eval", Intr("aau", [], "v", pure=False), guard, op.text.strip()))
            return
        # ---------------- unknown: keep as inline-asm-like intrinsic
        L.unknown.append(op.text.strip())
        srcs = []
        for s in a[:-1]:
            try:
                srcs.append(R(s, "i64"))
            except Exception:
                pass
        name, view = self.dst(a[-1], bn) if a else (None, None)
        if name:
            effects.append(("set", name, view if view != "b" else "b",
                            Intr(m, srcs, view if view else "i64", pure=False), guard))
        else:
            effects.append(("eval", Intr(m, srcs, "v", pure=False), guard, op.text.strip()))
