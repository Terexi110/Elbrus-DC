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
                 ExprStmt, is_float, BITS, map_expr, mask)
from .simplify import getf_expr

PACKED = {"paddw": 2, "psubw": 2, "pcmpgtw": 2, "pcmpeqw": 2, "pfadds": 2, "pfsubs": 2,
          "pfmuls": 2, "pshufb": 3}          # mnemonic -> number of sources

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
           "alc", "eap", "apb", "wait", "setei", "pref", "flushr", "flushc",
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
        self.loop = False      # bundle uses loop machinery we do not model (AAU, pcnt...)
        self.abn = None        # (abnf, abnt): rotate %b[] after this bundle
        self.alc = None        # (alcf, alct): advance loop counter
        self.abp = None        # (abpf, abpt): rotate %pred0..%pred<psz>
        self.setbp = None      # psz
        self.aa_ch = {}        # channel -> (aasti index, element bytes) of its staa/ldaa
        self.incr = []         # (channel, aaincr index): post-increment of that op's aasti
        self.apb_am = set()    # APB areas whose read pointer advances after this bundle


class Lifter:
    def __init__(self, func, resolver):
        self.func = func
        self.res = resolver    # symbol resolver (see decompiler.Resolver)
        self._scan_fapb()
        self._scan_loop_mode()

    def _scan_loop_mode(self):
        """Stores executed earlier in the same iteration of a loop-mode body, per
        bundle: at loop end each store decrements LSR.strmd, so the k-th store of
        the last iteration is enabled iff strmd > k.  Loop-mode bodies are
        straight lines from the disp target (header) to the bundle with ct."""
        bundles = sorted(getattr(self.func, "bundles", []), key=lambda b: b.addr)
        targets = set()
        for b in bundles:
            for o in b.ops:
                if o.mnem == "disp" and len(o.args) > 1:
                    t = parse_imm(o.args[1])[0]
                    if t is not None:
                        targets.add(t)
        self.lm_before = {}
        n = 0
        prev_ct = True
        for b in bundles:
            lm = any(o.mnem == "loop_mode" for o in b.ops)
            if not lm or prev_ct or b.addr in targets:
                n = 0
            self.lm_before[b.addr] = n
            if lm:
                n += min(2, sum(1 for o in b.ops if o.mnem in STORES or
                                re.match(r"^(staa[bhwdq]p?|stqp|stmqp)$", o.mnem)))
            prev_ct = any(o.mnem == "ct" for o in b.ops)

    # ------------------------------------------------------------ APB programs
    def _scan_fapb(self):
        """FAPB programs (bundles made of `apb` entries) and the `ldisp %ctpr2`
        that select them.  Entry i: low half -> area i, high half -> area 32+i
        (dpl=1: the high half duplicates the low one)."""
        FMT = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}
        self.fapb = {}          # program address -> {area: dict(fmt, d, ind, incr, disp, mrng)}
        self.ldisp2 = []        # (bundle addr, program addr)
        for b in getattr(self.func, "bundles", []):
            ents = [o for o in b.ops if o.mnem == "apb"]
            if ents:
                prog, i = {}, 0
                for k in range(0, len(ents), 2):
                    lo = dict(x.replace(" ", "").split("=", 1) for x in ents[k].args if "=" in x)
                    hi = dict(x.replace(" ", "").split("=", 1) for x in ents[k + 1].args if "=" in x) \
                        if k + 1 < len(ents) else {}

                    def area(f):
                        fmt = FMT.get(int(f.get("fmt", "0"), 0))
                        if not fmt:
                            return None
                        return {"fmt": fmt, "d": int(f.get("d", "0"), 0), "ind": int(f.get("ind", "0"), 0),
                                "incr": int(f.get("incr", "0"), 0), "disp": int(f.get("disp", "0"), 0),
                                "mrng": int(f.get("mrng", "0"), 0) or 32}
                    a0 = area(lo)
                    if a0:
                        prog[i] = a0
                    if hi.get("dpl", "0") == "1":
                        if a0:
                            prog[32 + i] = a0
                    else:
                        a1 = area(hi)
                        if a1:
                            prog[32 + i] = a1
                    i += 1
                    if lo.get("ct", "0") == "1":
                        break
                self.fapb[b.addr] = prog
            for o in b.ops:
                if o.mnem == "ldisp" and o.args and o.args[0].strip() == "%ctpr2":
                    t = parse_imm(o.args[1])[0] if len(o.args) > 1 else None
                    if t is not None:
                        self.ldisp2.append((b.addr, t))

    def fapb_at(self, addr):
        """Program selected by the last `ldisp %ctpr2` before addr (linear order)."""
        best = None
        for a, t in self.ldisp2:
            if a <= addr:
                best = t
        return self.fapb.get(best) if best is not None else None

    # ------------------------------------------------------------ operands
    def breg(self, idx, bn):
        rbs, rsz, rcur = bn[:3] if bn else (0, 0, 0)
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
                # the bundle's literal syllables are not in operand order, and two
                # operations may share one syllable (`addd 0, gtab` and `ldw 0, gtab`
                # with a single relocation): take the relocation whose in-place value
                # is the literal objdump printed, preferring one not used yet; without
                # the section bytes, the one whose addend matches, else the next one
                inplace = [e for e in lits if e[1] is not None and e[1] == val]
                if inplace:
                    e = next((x for x in inplace if not x[2]), None)
                    if e is None and len({x[0][2] for x in inplace}) == 1:
                        e = inplace[0]                  # shared literal
                if not inplace or e is None:
                    free = [x for x in lits if not x[2]] or lits
                    e = free[0]
                    for x in free:
                        m = re.match(r"^.+?([+-]0x[0-9a-f]+)?$", x[0][2].strip())
                        add = int(m.group(1), 16) if m and m.group(1) else 0
                        if add == val:
                            e = x
                            break
                e[2] = True
                return self.res.reloc_expr(e[0], val)
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
        elif s in ("%NOT_LOOP_END", "%LOOP_END"):
            e = Intr(s[1:].lower(), [], "b", pure=False)
        elif s == "%MLOCK":
            # MLOCK guards the re-computation from the re-loaded (check-unlock) value after
            # a store hit a locked speculative load.  Re-computing when nothing changed gives
            # the same result, so always taking that path is semantically exact.
            e = Const(1, "b")
        else:
            m = re.match(r"^%pcnt(\d+)$", s)
            e = Bin("<=u", Reg("pc", "i64"), Const(int(m.group(1)), "i64"), "b") if m else \
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
        lits = [[r, self.res.lit_inplace(bundle, r), False]
                for r in sorted(bundle.relocs, key=lambda r: r[0]) if "DISP" not in r[1]]
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
        ef_chan = {}           # effect index -> channel of the op that produced it
        for op in normal + plu:
            m = op.mnem
            a = [x for x in op.args if not x.startswith("mas")]
            guard = self.cond(op.guard, bn) if m not in ("ct", "ibranch", "call") else None
            n0 = len(effects)
            try:
                self._lift_op(L, op, m, a, guard, bn, lits, effects, lp, plu_writes)
            except Exception as ex:  # robust: never die on one op
                L.unknown.append(op.text.strip())
                effects.append(("eval", Intr("asm", [], "v", pure=False), None, op.text.strip()))
            for i in range(n0, len(effects)):
                ef_chan[i] = op.chan if op.chan is not None else 9
        if any(o.mnem == "loop_mode" for o in bundle.ops):
            self._loop_mode_stores(L, effects, ef_chan)
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
        self._loop_effects(L, bn)
        # prologue counter (%pcntN, and the store suppression while LSR.pcnt != 0)
        # and %MLOCK re-execution are not modelled: flag the function
        def opaque_loop(e):
            return any(isinstance(x, Intr) and x.name in ("pcnt", "mlock") for x in e.walk())
        if any(opaque_loop(e) for st in L.stmts for e in st.exprs()) or \
                (L.ct and L.ct[1] is not None and opaque_loop(L.ct[1])):
            L.loop = True
        return L

    def _loop_mode_stores(self, L, effects, ef_chan):
        """Stores in a loop-mode bundle (HS.lm) are enabled only when
            LSR.semc || (pcnt == 0 && !(loop_end && strmd <= st_op))
        where st_op is 0 for the first store channel of the bundle (stl), 1 for
        the second (str), and loop_end = vlc && lcnt < 2 && ecnt == 0 (qemu
        e2k_translate_calc_ld_st_loop_conds).  After a store at loop end
        strmd is decremented (if non-zero): for the bundles that follow in the
        same (last) iteration this is folded into st_op (see _scan_loop_mode)."""
        idx = sorted((ef_chan[i], i) for i, ef in enumerate(effects) if ef[0] == "store")
        if not idx:
            return
        R64 = lambda n: Reg(n, "i64")
        base = self.lm_before.get(L.bundle.addr, 0)
        for k, (_, i) in enumerate(idx[:2]):
            _, addr, val, ty, guard = effects[i]
            en = Intr("st_enbl", [R64("lsr_semc"), R64("pc"), R64("lsr_vlc"), R64("lc"),
                                  R64("lsr_strmd"), Const(base + k, "i64")], "b")
            effects[i] = ("store", addr, val, ty, en if guard is None else Bin("&&", guard, en, "b"))

    def _loop_effects(self, L, bn):
        """Hardware-loop state after the bundle: counter step, %b[] rotation,
        and %NOT_LOOP_END / %LOOP_END in the branch condition."""
        a = L.bundle.addr
        for ar in sorted(L.apb_am):
            r = Reg("aaldi%d" % ar, "i64")
            L.stmts.append(Assign(r, Bin("+", r, Reg("apbinc%d" % ar, "i64"), "i64"), addr=a))
        for ch, n in L.incr:
            if ch in L.aa_ch:
                k, nbytes = L.aa_ch[ch]
                r = Reg("aasti%s" % k, "i64")
                step = Const(nbytes, "i64") if n == 0 else \
                    Bin("*", Reg("aaincr%d" % n, "i64"), Const(nbytes, "i64"), "i64")
                L.stmts.append(Assign(r, Bin("+", r, step, "i64"), addr=a))
            else:
                L.loop = True
        lc = Reg("lc", "i64")
        if L.alc == (True, True):
            L.stmts.append(Assign(lc, Bin("-", lc, Const(1, "i64"), "i64"), addr=a))
            pc = Reg("pc", "i64")       # prologue counter: decremented while non-zero
            L.stmts.append(Assign(pc, Select(Bin("!=", pc, Const(0, "i64"), "b"),
                                             Bin("-", pc, Const(1, "i64"), "i64"), pc, "i64"), addr=a))
        # conditions are read after the statements: express them on the stepped counter
        stepped = L.alc == (True, True)

        def fix(e):
            if isinstance(e, Intr) and e.name in ("not_loop_end", "loop_end"):
                if e.name == "not_loop_end":
                    return Bin("!=", lc, Const(0, "i64"), "b") if stepped else Bin(">u", lc, Const(1, "i64"), "b")
                return Bin("==", lc, Const(0, "i64"), "b") if stepped else Bin("<=u", lc, Const(1, "i64"), "b")
            return e
        if L.abp == (True, True) and bn and len(bn) > 3 and bn[3] > 0:
            # rotating predicates: %pred[i+1] of the next iteration is %pred[i] of this one
            P = bn[3] + 1
            tmps = []
            for i in range(P):
                t = "prot%x_%d" % (a, i)
                L.stmts.append(Assign(Reg(t, "b"), Reg("pred%d" % i, "b"), addr=a))
                tmps.append(t)
            for i in range(P):
                L.stmts.append(Assign(Reg("pred%d" % ((i + 1) % P), "b"), Reg(tmps[i], "b"), addr=a))
        if L.abn == (True, True) and bn:
            # %b[i] of the next iteration is %b[i-2] of this one.  Names stay those of
            # the static (setbn) mapping, so shift the contents:  N(i) := N(i-2).
            rbs, rsz, rcur = bn[:3]
            R = 2 * (rsz + 1)
            names = [self.breg(i, bn) for i in range(R)]
            tmps = []
            for i in range(R):
                t = "rot%x_%d" % (a, i)
                L.stmts.append(Assign(Reg(t, "any"), Reg(names[i], "any"), addr=a))
                tmps.append(t)
            for i in range(R):
                L.stmts.append(Assign(Reg(names[(i + 2) % R], "any"), Reg(tmps[i], "any"), addr=a))
        if L.ct and L.ct[1] is not None:
            # ct reads the state *before* the bundle, but the branch condition is
            # evaluated after the block's statements: snapshot registers the bundle
            # overwrites (rotation, same-bundle writes).  `lc` is already in post form.
            written = {st.dst.name for st in L.stmts if isinstance(st, Assign) and isinstance(st.dst, Reg)}
            snap = {}

            def pre(e):
                if isinstance(e, Reg) and e.name in written and e.name != "lc":
                    if e.name not in snap:
                        snap[e.name] = Reg("ctc%x_%s" % (a, e.name), e.ty)
                    return snap[e.name]
                return e
            c = map_expr(L.ct[1], pre)
            L.stmts[0:0] = [Assign(t, Reg(n, t.ty), addr=a) for n, t in snap.items()]
            L.ct = (L.ct[0], map_expr(c, fix))
        if L.call and L.call[2] is not None:
            # the same for a predicated call: `call ... ? %pred2` with a PLU write of %pred2
            # in the bundle uses the old value (the call is appended after the statements)
            written = {st.dst.name for st in L.stmts if isinstance(st, Assign) and isinstance(st.dst, Reg)}
            snap = {}

            def pre2(e):
                if isinstance(e, Reg) and e.name in written:
                    if e.name not in snap:
                        snap[e.name] = Reg("ccc%x_%s" % (a, e.name), e.ty)
                    return snap[e.name]
                return e
            c = map_expr(L.call[2], pre2)
            L.stmts[0:0] = [Assign(t, Reg(n, t.ty), addr=a) for n, t in snap.items()]
            L.call = (L.call[0], L.call[1], c)

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
            if m == "setbp":
                d = dict(re.findall(r"(\w+) = (0x[0-9a-f]+)", " ".join(op.args)))
                L.setbp = int(d.get("psz", "0"), 16)
            if m in ("abn", "alc", "abp"):
                d = dict(re.findall(r"(\w+)=(\d)", " ".join(op.args)))
                f, t = d.get(m + "f", "1") == "1", d.get(m + "t", "1") == "1"
                setattr(L, m, (f, t))
                if not (f and t):
                    L.loop = True          # conditional rotation/count: not modelled
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
            # qemu e2k_2stage_*: res = func2(src3, src1 op1 src2), with
            # sub(a, b) = a - b and rsub(a, b) = b - a, so
            #   op1_sub  = src3 - (src1 op1 src2)
            #   op1_rsub = (src1 op1 src2) - src3
            if o2 == "rsub":
                e = Bin("-", r1, z, ty)
            elif o2 == "sub":
                e = Bin("-", z, r1, ty)
            else:
                e = Bin(INT_BIN[o2.lstrip("r")], z, r1, ty)
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
        mm = re.match(r"^getf_(r?sub|add|andn?|orn?|xorn?)([sd])$", m)
        if mm and len(a) == 4:
            # qemu e2k_2stage_getf_*: res = func(src3, getf(src1, src2))
            ty = {"s": "i32", "d": "i64"}[mm.group(2)]
            v, _ = parse_imm(a[1])
            src = R(a[0], ty)
            g = getf_expr(src, v, ty) if v is not None else Intr("getf" + mm.group(2), [src, R(a[1], ty)], ty)
            z = R(a[2], ty)
            o = mm.group(1)
            if o == "rsub":
                e = Bin("-", g, z, ty)
            elif o == "sub":
                e = Bin("-", z, g, ty)
            elif o == "add":
                e = Bin("+", z, g, ty)
            elif o.endswith("n") and o != "and":
                base = {"andn": "&", "orn": "|", "xorn": "^"}[o]
                e = Un("~", Bin("^", z, g, ty), ty) if o == "xorn" else Bin(base, z, Un("~", g, ty), ty)
            else:
                e = Bin({"and": "&", "or": "|", "xor": "^"}[o], z, g, ty)
            setr(a[3], e, ty)
            return
        if m in ("getfs", "getfd"):
            ty = "i32" if m == "getfs" else "i64"
            bits = 32 if ty == "i32" else 64
            v, _ = parse_imm(a[1])
            src = R(a[0], ty)
            if v is None:
                setr(a[2], Intr(m, [src, R(a[1], ty)], ty), ty)
                return
            setr(a[2], getf_expr(src, v, ty), ty)
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
            e = Load(addr, ty, spec="sm" in op.flags)
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
            mas = [x for x in op.args if x.strip().startswith("mas")]
            if mas:
                mv = re.findall(r"0x[0-9a-f]+", mas[0])
                if mv and int(mv[0], 16) & 7 == 7:
                    return              # MAS.mod == 7: MMU/AAU control, not a data store
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
            elf = getattr(self.res, "elf", None)
            if a[0].strip() == "%ip" and elf is not None and not elf.is_rel:
                # %ip reads the address of this wide instruction (qemu: sregs.ip):
                # position-independent code adds an offset to reach .got / data
                setr(a[1], Const(L.bundle.addr, ty), ty)
                return
            setr(a[1], Intr("rr", [Intr("sr_" + a[0].strip("%"), [], ty)], ty, pure=False), ty)
            return
        if m == "rwd" and a[1].strip() == "%lsr":
            # LSR: lcnt[31:0], ecnt[36:32], vlc[37].  The kernel runs lcnt+ecnt times
            # (vlc=1); model that as one down-counter `lc` (see alc / NOT_LOOP_END).
            x = R(a[0], "i64")
            lcnt = Bin("&", x, Const(0xffffffff, "i64"), "i64")
            ecnt = Bin("&", Bin(">>u", x, Const(32, "i64"), "i64"), Const(0x1f, "i64"), "i64")
            effects.append(("set", "lc", "i64", Bin("+", lcnt, ecnt, "i64"), guard))
            pcnt = Bin("&", Bin(">>u", x, Const(48, "i64"), "i64"), Const(0x1f, "i64"), "i64")
            effects.append(("set", "pc", "i64", pcnt, guard))
            # store enabling in loop mode: vlc[37], strmd[59:53], semc[60]
            for nm, sh, mk in (("lsr_vlc", 37, 1), ("lsr_strmd", 53, 0x7f), ("lsr_semc", 60, 1)):
                f = Bin("&", Bin(">>u", x, Const(sh, "i64"), "i64"), Const(mk, "i64"), "i64")
                effects.append(("set", nm, "i64", f, guard))
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
        if m in ("insfd", "insfs") and len(a) == 4 and parse_imm(a[1])[0] is None:
            # field spec in a register: same formula with variable shift/size
            ty, bits = ("i64", 64) if m == "insfd" else ("i32", 32)
            x, sp, z = R(a[0], ty), R(a[1], ty), R(a[2], ty)
            sh = Bin("&", sp, Const(bits - 1, ty), ty)
            size = Bin("&", Bin(">>u", sp, Const(6, ty), ty), Const(bits - 1, ty), ty)
            hi = Bin("<<", Const(mask(bits), ty), size, ty)
            e = Bin("|", Bin("&", Bin("rotr", x, sh, ty), hi, ty),
                    Bin("&", z, Un("~", hi, ty), ty), ty)
            setr(a[3], e, ty)
            return
        if m in ("insfd", "insfs") and len(a) == 4 and parse_imm(a[1])[0] is not None:
            # qemu: res = (rotr(src1, sh) & (~0 << size)) | (src3 & ~(~0 << size))
            ty, bits = ("i64", 64) if m == "insfd" else ("i32", 32)
            v = parse_imm(a[1])[0]
            sh, size = v & (bits - 1), (v >> 6) & (bits - 1)
            hi = (mask(bits) << size) & mask(bits)
            x, z = R(a[0], ty), R(a[2], ty)
            rot = Bin("rotr", x, Const(sh, ty), ty) if sh else x
            e = Bin("|", Bin("&", rot, Const(hi, ty), ty), Bin("&", z, Const(~hi & mask(bits), ty), ty), ty)
            setr(a[3], e, ty)
            return
        # ---------------- packed (SIMD within a 64-bit register), see qemu packed_helper.c
        if m in PACKED and len(a) == PACKED[m] + 1:
            srcs = [R(x, "i64") for x in a[:-1]]
            setr(a[-1], Intr(m, srcs, "i64", pure=True), "i64")
            return
        # ---------------- AAU: array access unit (direct form, no APB)
        if m in ("aaurws", "aaurwd", "aaurwq"):
            mm = re.match(r"^%aa(d|sti|incr|ind)(\d+)$", a[1].strip())
            if mm:
                ty = "i32" if m == "aaurws" else "i64"
                if m == "aaurwq":
                    # qemu: AAD.lo = low 64 bits of the quad (base), AAD.hi = 0
                    name, _ = self.reg_of(a[0].strip(), bn)
                    x = Reg(name, "i64") if name else R(a[0], "i64")
                else:
                    x = R(a[0], ty)
                if ty == "i32":
                    x = Cast(x, "i64", signed=False)
                if mm.group(1) == "incr" and mm.group(2) == "0":
                    return                         # AAINCR0 is always 1
                effects.append(("set", "aa%s%s" % mm.groups(), "i64", x, guard))
                return
            if a[1].strip().startswith("%aabf"):
                return                             # buffer format/flush: no data effect
        mm = re.match(r"^(staa|ldaa)([bhwd])$", m)
        if mm:
            arg = a[1] if mm.group(1) == "staa" else a[0]
            am = re.match(r"^%aad(\d+)\s*\[\s*%aasti(\d+)\s*(?:\+\s*(.*?))?\s*\]$", arg.strip())
            if am:
                ty = {"b": "i8", "h": "i16", "w": "i32", "d": "i64"}[mm.group(2)]
                nbytes = BITS[ty] // 8
                lit = 0
                if am.group(3):
                    hx = re.findall(r"0x[0-9a-f]+", am.group(3))
                    lit = int(hx[-1], 16) if hx else 0
                addr = Bin("+", Reg("aad%s" % am.group(1), "i64"), Reg("aasti%s" % am.group(2), "i64"), "i64")
                if lit:
                    addr = Bin("+", addr, Const(lit, "i64"), "i64")
                if op.chan is not None:
                    L.aa_ch[op.chan] = (am.group(2), nbytes)
                if mm.group(1) == "staa":
                    val = R(a[0], "i32" if ty in ("i8", "i16") else ty)
                    if ty in ("i8", "i16"):
                        val = Cast(val, ty, signed=False)
                    effects.append(("store", addr, val, ty, guard))
                else:
                    e = Load(addr, ty)
                    if ty in ("i8", "i16"):
                        e = Cast(e, "i32", signed=False)
                    setr(a[1], e, "i32" if ty in ("i8", "i16") else ty)
                return
        if m == "incr" and op.chan is not None:
            mm = re.match(r"^%aaincr(\d+)$", a[0].strip())
            if mm:
                L.incr.append((op.chan, int(mm.group(1))))
                return
        # ---------------- APB: bap starts the prefetch program, mova* reads from it
        if m == "bap":
            prog = self.fapb_at(L.bundle.addr)
            if prog is None:
                L.loop = True
                return
            for ar, f in sorted(prog.items()):
                base = Reg("aad%d" % f["d"], "i64")
                inc = Const(1, "i64") if f["incr"] == 0 else Reg("aaincr%d" % f["incr"], "i64")
                idx = Const(0, "i64") if f["ind"] == 0 else Reg("aaind%d" % f["ind"], "i64")
                effects.append(("set", "apbbase%d" % ar, "i64", base, None))
                effects.append(("set", "apbinc%d" % ar, "i64", Bin("*", inc, Const(f["fmt"], "i64"), "i64"), None))
                effects.append(("set", "aaldi%d" % ar, "i64",
                                Bin("+", idx, Const(f["disp"], "i64"), "i64") if f["disp"] else idx, None))
            return
        mm = re.match(r"^mova([bhwd])$", m)
        if mm and op.chan is not None:
            d = dict(x.replace(" ", "").split("=", 1) for x in a[:-1] if "=" in x)
            ar = int(d.get("area", "0"), 0) + (32 if op.chan >= 2 else 0)
            ind = int(d.get("ind", "0"), 0)
            ty = {"b": "i8", "h": "i16", "w": "i32", "d": "i64"}[mm.group(1)]
            addr = Bin("+", Reg("apbbase%d" % ar, "i64"), Reg("aaldi%d" % ar, "i64"), "i64")
            if ind:
                addr = Bin("+", addr, Const(ind, "i64"), "i64")
            e = Load(addr, ty)
            if ty in ("i8", "i16"):
                e = Cast(e, "i32", signed=False)
            setr(a[-1], e, "i32" if ty in ("i8", "i16") else ty)
            if d.get("am", "0") == "1":
                L.apb_am.add(ar)
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
