"""Driver: binary -> pseudo-C."""
import re

from . import objdump as od
from .elfread import Elf, STT_FUNC, STT_SECTION, SHT_NOBITS
from .ir import (Const, FConst, Var, Sym, Str, Bin, Un, Cast, Load, Call, Intr, Assign, Store, Select,
                 ExprStmt, Return, Branch, IJump, Switch, map_expr, is_int, is_float, BITS, sext)
from . import cfg as cfgmod
from . import analysis as an
from .analysis import VarInfo
from .simplify import simplify
from .structure import structure
from .emit import Printer, HEADER, SIGNED, UNSIGNED, esc

SKIP = {"_init", "_fini", "_start", "_dl_relocate_static_pie", "deregister_tm_clones",
        "register_tm_clones", "__do_global_dtors_aux", "frame_dummy", "__libc_csu_init",
        "__libc_csu_fini", "@plt_header"}
VARIADIC = {"printf": "int printf(const char *, ...);", "fprintf": "int fprintf(void *, const char *, ...);",
            "sprintf": "int sprintf(char *, const char *, ...);",
            "snprintf": "int snprintf(char *, unsigned long, const char *, ...);",
            "scanf": "int scanf(const char *, ...);", "sscanf": "int sscanf(const char *, const char *, ...);",
            "fscanf": "int fscanf(void *, const char *, ...);", "open": "int open(const char *, int, ...);",
            "ioctl": "int ioctl(int, unsigned long, ...);", "fcntl": "int fcntl(int, int, ...);",
            "syslog": "void syslog(int, const char *, ...);", "err": "void err(int, const char *, ...);",
            "errx": "void errx(int, const char *, ...);", "warn": "void warn(const char *, ...);",
            "warnx": "void warnx(const char *, ...);", "dprintf": "int dprintf(int, const char *, ...);",
            "asprintf": "int asprintf(char **, const char *, ...);", "execl": "int execl(const char *, const char *, ...);",
            "execlp": "int execlp(const char *, const char *, ...);"}


def parse_reloc_target(s):
    m = re.match(r"^(.+?)([+-]0x[0-9a-f]+)?$", s.strip())
    name = m.group(1)
    add = int(m.group(2), 16) if m.group(2) else 0
    return name, add


def cname(name):
    n = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if n and n[0].isdigit():
        n = "_" + n
    return n


class Resolver:
    def __init__(self, elf, funcs):
        self.elf = elf
        self.funcs = funcs
        self.by_addr = {}
        self.by_name = {}
        for i, f in enumerate(funcs):
            end = f.bundles[-1].addr + f.bundles[-1].size if f.bundles else f.addr
            if i + 1 < len(funcs) and funcs[i + 1].section == f.section:
                end = max(end, funcs[i + 1].addr) if funcs[i + 1].addr > f.addr else end
            f.end = end
            self.by_addr[(f.section, f.addr)] = f
            self.by_name[f.name] = f
        self.current = None

    @staticmethod
    def clean_func_name(n):
        for suf in ("@plt", "@secondary_plt"):
            if n.endswith(suf):
                return n[:-len(suf)]
        return n

    def func_at(self, section, addr):
        f = self.by_addr.get((section, addr))
        return self.clean_func_name(f.name) if f else None

    def disp_target(self, bundle, text):
        func = bundle.func
        rels = [r for r in bundle.relocs if "DISP" in r[1]]
        if rels:
            off, typ, tgt = rels[0]
            name, add = parse_reloc_target(tgt)
            delta = add - (off - bundle.addr)
            if name.startswith("."):
                if name == func.section and func.addr <= delta < func.end:
                    return ("label", delta)
                f = self.func_at(name, delta)
                return ("func", f) if f else ("func", "%s_%x" % (cname(name), delta))
            return ("func", self.clean_func_name(name))
        try:
            addr = int(text, 16)
        except ValueError:
            return ("unknown",)
        if func.addr <= addr < func.end:
            return ("label", addr)
        f = self.func_at(func.section, addr)
        if f:
            return ("func", f)
        return ("abs", addr)

    def reloc_expr(self, rel, val):
        off, typ, tgt = rel
        name, add = parse_reloc_target(tgt)
        kind = "data"
        f = self.by_name.get(name)
        if f is not None or ("FUNC" in typ):
            kind = "func"
        for s in self.elf.symbols if self.elf else []:
            if s.name == name and s.type == STT_FUNC:
                kind = "func"
                break
        e = Sym(name, add + (val if val and not add else 0), kind)
        if "GOT" in typ and "GOTOFF" not in typ:
            e.got = True
        return e


class Context:
    """Global information used while printing (symbols, strings, declarations)."""

    def __init__(self, elf, resolver):
        self.elf = elf
        self.res = resolver
        self.globals = {}       # name -> {"types": set, "array": bool, "addr": bool}
        self.protos = {}        # called function -> (ret, [arg types])
        self.defined = {}       # function name -> signature

    # ---- symbols
    def _sym_location(self, sym):
        """(section, offset) for Sym."""
        if not self.elf:
            return None, None
        if self.elf.section(sym.name) is not None:
            return self.elf.section(sym.name), sym.off
        for s in self.elf.symbols:
            if s.name == sym.name and 0 < s.shndx < len(self.elf.sections) and s.type != STT_SECTION:
                sec = self.elf.sections[s.shndx]
                base = s.value - (sec.addr if not self.elf.is_rel else 0)
                return sec, base + sym.off
        return None, None

    def normalize(self, sym):
        """Map section-relative symbols to the named data symbol covering them."""
        if not self.elf:
            return sym
        sec = self.elf.section(sym.name)
        if sec is None:
            return sym
        addr = sym.off + (sec.addr if not self.elf.is_rel else 0)
        best = self.elf.symbol_at(sec.idx, addr)
        if best is not None:
            return Sym(best.name, addr - best.value, "func" if best.type == STT_FUNC else "data")
        return sym

    def string_at(self, sym):
        sec, off = self._sym_location(sym)
        if sec is None or not (sec.name.startswith(".rodata") or sec.name.startswith(".str")
                               or sec.name == ".data.rel.ro"):
            return None
        s = self.elf.cstring(sec, off)
        if s is None or (len(s) == 0):
            return None
        return s

    def jump_table(self, sym, func, limit=None):
        """Targets of a jump table starting at Sym (list of addresses) or None."""
        if not self.elf:
            return None
        sym = self.normalize(sym)
        sec, off = self._sym_location(sym)
        if sec is None:
            return None
        count = limit or 1024
        for s in self.elf.symbols:
            if s.name == sym.name and s.size and sym.off == 0 and s.type != STT_SECTION:
                count = min(count, s.size // 8)
                break
        out = []
        if self.elf.is_rel:
            rel = {r.offset: r for r in self.elf.relocs.get(sec.idx, [])}
            for k in range(count):
                r = rel.get(off + 8 * k)
                if r is None or r.sym is None:
                    break
                if r.sym.type == STT_SECTION:
                    if self.elf.sections[r.sym.shndx].name != func.section:
                        break
                    t = r.addend
                else:
                    t = r.sym.value + r.addend
                if not (func.addr <= t < func.end):
                    break
                out.append(t)
        else:
            for k in range(count):
                raw = self.elf.read(sec, off + 8 * k, 8)
                if raw is None:
                    break
                t = int.from_bytes(raw, "little")
                if not (func.addr <= t < func.end):
                    break
                out.append(t)
        return out or None

    def note_global(self, name, ty=None, array=False, addr=False):
        g = self.globals.setdefault(name, {"types": set(), "array": False, "addr": False})
        if ty:
            g["types"].add(ty)
        g["array"] |= array
        g["addr"] |= addr

    def local_info(self, sym, F):
        for L in getattr(F, "locals", []):
            if L["name"] == sym.name:
                return L
        return None

    def global_access(self, sym, ty, F):
        sym = self.normalize(sym) if sym.kind == "data" else sym
        if sym.kind == "local":
            L = self.local_info(sym, F)
            if L is None:
                return None
            esz = BITS[L["ty"]] // 8
            if L["count"] == 1 and sym.off == 0 and L["ty"] == ty:
                return L["name"]
            if L["ty"] == ty and sym.off % esz == 0 and L["count"] > 1:
                return "%s[%d]" % (L["name"], sym.off // esz)
            return "*(%s *)((char *)&%s + %d)" % (SIGNED.get(ty, "long"), L["name"], sym.off)
        n = cname(sym.name)
        if sym.off == 0:
            self.note_global(sym.name, ty)
            return n
        self.note_global(sym.name, None, array=True)
        return "*(%s *)((char *)&%s + %s)" % (SIGNED.get(ty, "long"), n, sym.off)

    def array_name(self, sym, ty, F):
        sym = self.normalize(sym) if sym.kind == "data" else sym
        if sym.kind == "local":
            L = self.local_info(sym, F)
            if L and L["ty"] == ty and sym.off == 0:
                return L["name"]
            return None
        if sym.off == 0:
            self.note_global(sym.name, ty, array=True)
            return cname(sym.name)
        return None

    def sym_value(self, x, F):
        if x.kind == "func":
            return (cname(x.name), 16, "void *")
        if x.kind == "label":
            return ("&&" + x.name, 14, "void *")
        if x.kind == "local":
            L = self.local_info(x, F)
            if L:
                if L["count"] > 1:
                    if x.off == 0:
                        return (L["name"], 16, "char *")
                    esz = BITS[L["ty"]] // 8
                    if x.off % esz == 0:
                        return ("&%s[%d]" % (L["name"], x.off // esz), 14, "char *")
                if x.off == 0:
                    return ("&" + L["name"], 14, "char *")
                return ("(char *)&%s + %d" % (L["name"], x.off), 12, "char *")
        s = self.string_at(x)
        if s is not None:
            return (esc(s), 16, "char *")
        x = self.normalize(x)
        if x.kind == "func":
            return (cname(x.name), 16, "void *")
        self.note_global(x.name, addr=True)
        n = cname(x.name)
        if x.off == 0:
            return ("&" + n, 14, "char *")
        return ("(char *)&%s + %s" % (n, x.off), 12, "char *")

    def declarations(self, funcs_defined):
        out = []
        for name, g in sorted(self.globals.items()):
            n = cname(name)
            tys = g["types"]
            if len(tys) == 1 and not g["array"]:
                out.append("extern %s %s;" % (SIGNED.get(next(iter(tys)), "long"), n))
            elif len(tys) == 1 and g["array"]:
                out.append("extern %s %s[];" % (SIGNED.get(next(iter(tys)), "long"), n))
            else:
                out.append("extern char %s[];" % n)
        for name, (ret, args) in sorted(self.protos.items()):
            if name in funcs_defined:
                continue
            if name in VARIADIC:
                out.append(VARIADIC[name])
                continue
            out.append("%s %s(%s);" % (ret, cname(name), ", ".join(args) or "void"))
        return out


# =================================================================== per function
def lift_function(func, resolver, summaries, jtables=None):
    F = cfgmod.build(func, resolver, jtables)
    if not F.blocks:
        return None
    an.decide_void(F)
    nparams_hint = summaries.get(func.name, {}).get("nparams") if summaries else None
    an.assign_call_args(F, summaries or {}, fparams=nparams_hint)
    for blk, st in F.calls:
        tn = st.src.target.name if isinstance(st.src.target, Sym) else None
        if tn and summaries.get(tn, {}).get("void"):
            st.src.ret_used = False
    an.build_vars(F)
    return F


def optimize(F):
    an.simplify_all(F)
    an.merge_selects(F)
    an.propagate(F)
    an.dce(F)
    an.merge_selects(F)
    an.propagate(F)
    an.dce(F)


def table_index(addr):
    """Load address pattern  Sym + idx*8  ->  (Sym, idx)"""
    if not (isinstance(addr, Bin) and addr.op == "+"):
        return None
    for a, b in ((addr.a, addr.b), (addr.b, addr.a)):
        if isinstance(a, Sym):
            if isinstance(b, Bin) and b.op == "<<" and isinstance(b.b, Const) and b.b.value == 3:
                return a, b.a
            if isinstance(b, Bin) and b.op == "*" and isinstance(b.b, Const) and b.b.value == 8:
                return a, b.a
    return None


def find_jump_tables(F, ctx):
    found = {}
    for b in F.block_list():
        t = b.term
        if isinstance(t, IJump) and isinstance(t.target, Load):
            m = table_index(t.target.addr)
            if m:
                targets = ctx.jump_table(m[0], F.func)
                if targets:
                    found[t.at] = targets
    return found


def to_switch(F):
    for b in F.block_list():
        t = b.term
        if isinstance(t, IJump) and getattr(t, "targets", None) and isinstance(t.target, Load):
            m = table_index(t.target.addr)
            if m:
                idx = m[1]
                if isinstance(idx, Cast) and idx.signed and idx.a.ty == "i32":
                    idx = idx.a
                b.term = Switch(idx, t.targets)
    cfgmod.link(F)


def summarize(F):
    params = [v for v in F.vars if v.is_param and has_uses(F, v)]
    n = max([v.param_index + 1 for v in params], default=0)
    return {"nparams": n, "void": not F.returns_value}


def has_uses(F, v):
    defs, uses = an.collect(F)
    return bool(uses.get(v))


# ------------------------------------------------------------------ finalize passes
def resolve_exe_consts(F, ctx):
    elf = ctx.elf
    if not elf or elf.is_rel:
        return

    def fix(e):
        if isinstance(e, Const) and e.ty == "i64" and e.value >= 0x1000:
            sec = elf.section_of_addr(e.value)
            if sec is not None and sec.name not in (".text",):
                sym = elf.symbol_at(sec.idx, e.value)
                if sym is not None and sym.name:
                    return Sym(sym.name, e.value - sym.value, "func" if sym.type == STT_FUNC else "data")
                return Sym(sec.name, e.value - sec.addr, "data")
            if sec is not None and sec.name == ".text":
                f = ctx.res.func_at(".text", e.value)
                if f:
                    return Sym(f, 0, "func")
        return e

    for b in F.block_list():
        for s in b.stmts:
            s.map(lambda x: map_expr(x, fix))
        if b.term is not None:
            b.term.map(lambda x: map_expr(x, fix))


def sp_offset(e, spv):
    """e == sp + c -> c ; else None"""
    if isinstance(e, Var) and e.var is spv:
        return 0
    if isinstance(e, Bin) and e.op == "+" and isinstance(e.a, Var) and e.a.var is spv and isinstance(e.b, Const):
        return sext(e.b.value, 64)
    return None


def layout_frame(F):
    spv = next((v for v in F.vars if v.reg == "sp"), None)
    F.locals = []
    F.stack_params = {}
    if spv is None:
        return
    N = F.frame
    slots = {}      # off -> set(types)
    escapes = set()

    indexed = {}

    def idx_off(a):
        if isinstance(a, Bin) and a.op == "+" and isinstance(a.b, Const) and isinstance(a.a, Bin) \
                and a.a.op == "+" and isinstance(a.a.a, Var) and a.a.a.var is spv:
            return sext(a.b.value, 64)
        return None

    def scan(e, direct=False):
        if isinstance(e, Load):
            c = sp_offset(e.addr, spv)
            if c is not None:
                slots.setdefault(c, set()).add(e.ty)
                return
            if idx_off(e.addr) is not None:
                scan(e.addr.a.b)
                return
            scan(e.addr)
            return
        c = sp_offset(e, spv)
        if c is not None:
            escapes.add(c)
            return
        for ch in e.children():
            scan(ch)

    for b in F.block_list():
        for s in b.stmts:
            for e in s.exprs():
                for x in e.walk():
                    if isinstance(x, Load) and idx_off(x.addr) is not None:
                        indexed.setdefault(idx_off(x.addr), set()).add(x.ty)
            if isinstance(s, Store) and idx_off(s.addr) is not None:
                indexed.setdefault(idx_off(s.addr), set()).add(s.ty)
        if b.term is not None:
            for e in b.term.exprs():
                for x in e.walk():
                    if isinstance(x, Load) and idx_off(x.addr) is not None:
                        indexed.setdefault(idx_off(x.addr), set()).add(x.ty)

    for b in F.block_list():
        for s in b.stmts:
            if isinstance(s, Store):
                c = sp_offset(s.addr, spv)
                if c is not None:
                    slots.setdefault(c, set()).add(s.ty)
                elif idx_off(s.addr) is not None:
                    scan(s.addr.a.b)
                else:
                    scan(s.addr)
                scan(s.val)
                if s.guard is not None:
                    scan(s.guard)
            else:
                for e in s.exprs():
                    scan(e)
        if b.term is not None:
            for e in b.term.exprs():
                scan(e)
    # incoming stack parameters: loads at sp + 8*k, k >= 8
    params = {}
    for c, tys in slots.items():
        if c >= 0x40 and c % 8 == 0:
            params[c] = tys
    # locals: negative offsets
    loc_offs = sorted(set([c for c in slots if c < 0] + [c for c in escapes if c < 0] +
                          [c for c in indexed if c < 0]))
    if N == 0 and loc_offs:
        N = -loc_offs[0]
    locals_ = []
    bounds = loc_offs + [0]
    for i, c in enumerate(loc_offs):
        nxt = bounds[i + 1]
        tys = slots.get(c, set())
        arr_tys = set()
        if c in escapes:
            # look for element type through indexed accesses
            for b in F.block_list():
                for s in b.stmts:
                    for e in s.exprs() + ([s.addr] if isinstance(s, Store) else []):
                        for x in e.walk():
                            if isinstance(x, Load) and isinstance(x.addr, Bin) and x.addr.op == "+" and \
                                    sp_offset(x.addr.a, spv) == c:
                                arr_tys.add(x.ty)
                    if isinstance(s, Store) and isinstance(s.addr, Bin) and s.addr.op == "+" and \
                            sp_offset(s.addr.a, spv) == c:
                        arr_tys.add(s.ty)
        arr_tys |= indexed.get(c, set())
        ty = None
        if len(tys) == 1:
            ty = next(iter(tys))
        elif len(arr_tys) == 1 and not tys:
            ty = next(iter(arr_tys))
        elif arr_tys and tys and len(tys | arr_tys) == 1:
            ty = next(iter(tys))
        if ty is None:
            ty = "i8"
        esz = BITS.get(ty, 64) // 8
        span = nxt - c
        count = 1
        if arr_tys or (c in escapes and span > esz and not tys):
            count = max(1, span // esz)
        elif span < esz:
            ty, esz, count = "i8", 1, max(span, 1)
        if ty == "i128":
            ty, esz, count = "i64", 8, max(1, span // 8)
        locals_.append({"name": "local_%x" % (-c), "off": c, "ty": ty, "count": count, "size": span})
    F.locals = locals_

    def which_local(c):
        for L in locals_:
            if L["off"] <= c < L["off"] + L["size"]:
                return L
        return None

    pvars = {}

    def fix(e):
        if isinstance(e, Load):
            c = sp_offset(e.addr, spv)
            if c is not None and c >= 0x40 and c % 8 == 0:
                k = c // 8
                v = pvars.get(k)
                if v is None:
                    v = VarInfo("a%d" % (k + 1), "stack%d" % k, e.ty if e.ty in ("i32", "i64", "f32", "f64") else "i64")
                    v.is_param = True
                    v.is_input = True
                    v.param_index = k
                    pvars[k] = v
                    F.vars.append(v)
                return Var(v, v.ty) if v.ty == e.ty else Cast(Var(v, v.ty), e.ty, signed=False)
        c = sp_offset(e, spv)
        if c is not None and c < 0:
            L = which_local(c)
            if L:
                return Sym(L["name"], c - L["off"], "local")
        if c is not None and 0 <= c < 0x40:
            return Bin("+", Intr("va_area", [], "i64"), Const(c, "i64"), "i64")
        return e

    def fix_indexed(e):
        # (sp + X) + c  ->  &local + X
        if isinstance(e, Bin) and e.op == "+" and isinstance(e.b, Const) and isinstance(e.a, Bin) \
                and e.a.op == "+" and isinstance(e.a.a, Var) and e.a.a.var is spv:
            c = sext(e.b.value, 64)
            L = which_local(c)
            if L:
                return Bin("+", Sym(L["name"], c - L["off"], "local"), topdown(e.a.b), "i64")
        return None

    def topdown(e):
        r = fix_indexed(e)
        if r is not None:
            return r
        r = fix(e)
        if r is not e:
            return r
        return e.replace(topdown)

    for b in F.block_list():
        for s in b.stmts:
            s.map(topdown)
        if b.term is not None:
            b.term.map(topdown)
    F.stack_params = pvars


def fix_outgoing_stack_args(F):
    """Remove arg-homing stores before calls; attach args 9+ passed on stack."""
    spv = next((v for v in F.vars if v.reg == "sp"), None)
    def find_call(x):
        for e in x.exprs():
            for y in e.walk():
                if isinstance(y, Call):
                    return y
        return None

    for b in F.block_list():
        i = 0
        while i <= len(b.stmts):
            s = b.stmts[i] if i < len(b.stmts) else b.term
            if s is None:
                break
            call = s.src if isinstance(s, Assign) else s.e if isinstance(s, ExprStmt) else None
            if not isinstance(call, Call):
                call = find_call(s) if i == len(b.stmts) else None
            if not isinstance(call, Call):
                i += 1
                continue
            base = -F.frame
            extra = {}
            j = i - 1
            while j >= 0:
                st = b.stmts[j]
                if isinstance(st, (Assign, ExprStmt)) and any(isinstance(x, Call) for e in st.exprs() for x in e.walk()):
                    break
                if isinstance(st, Store) and spv is not None:
                    c = sp_offset(st.addr, spv)
                    if c is not None and c >= base and (c - base) % 8 == 0 and (c - base) < 0x40 + 8 * 32:
                        k = (c - base) // 8
                        if k >= 8:
                            extra[k] = st.val
                        del b.stmts[j]
                        i -= 1
                j -= 1
            if extra:
                n = max(extra) + 1
                args = list(call.args) + [Const(0, "i64")] * (8 - len(call.args))
                for k in range(8, n):
                    args.append(extra.get(k, Const(0, "i64")))
                call.args = args
            i += 1


def finalize_types(F, ctx, summaries):
    # call arguments: drop sign extensions of int arguments
    def fixcall(e):
        if isinstance(e, Call):
            e.args = [a.a if (isinstance(a, Cast) and a.signed and a.a.ty == "i32" and a.ty == "i64") else a
                      for a in e.args]
        return e

    for b in F.block_list():
        for s in b.stmts:
            s.map(lambda x: map_expr(x, fixcall))
        if b.term is not None:
            b.term.map(lambda x: map_expr(x, fixcall))
    # return type
    rets = [b.term for b in F.block_list() if isinstance(b.term, Return)]
    vals = [r.value for r in rets if r.value is not None]
    rty = "v"
    if vals:
        if all(isinstance(v, Cast) and v.a.ty == "i32" and v.ty == "i64" for v in vals):
            rty = "i32"
            for r in rets:
                if r.value is not None:
                    r.value = r.value.a
        elif all(v.ty in ("i32",) for v in vals):
            rty = "i32"
        elif any(is_float(v.ty) for v in vals):
            rty = next(v.ty for v in vals if is_float(v.ty))
        elif all(isinstance(v, Const) and -2**31 <= sext(v.value, 64) < 2**31 for v in vals):
            rty = "i32"
            for r in rets:
                r.value = Const(r.value.value, "i32")
        else:
            rty = "i64"
            if all(isinstance(v, Cast) and v.a.ty in ("i8", "i16") and v.ty == "i64" for v in vals):
                rty = vals[0].a.ty
                for r in rets:
                    if r.value is not None:
                        r.value = r.value.a
    F.ret_ty = rty
    # pointer-like variables (cosmetic char *)
    mark_pointers(F)
    # prototypes for called functions
    for b in F.block_list():
        for s in b.stmts:
            for e in s.exprs():
                for x in e.walk():
                    if isinstance(x, Call) and isinstance(x.target, Sym) and x.target.kind == "func":
                        name = x.target.name
                        if name not in ctx.protos:
                            ret = SIGNED.get(x.ty, "long") if x.ret_used else "long"
                            if summaries.get(name, {}).get("void"):
                                ret = "void"
                            ctx.protos[name] = (ret, [SIGNED.get(a.ty, "long") if not a.ty == "b" else "int" for a in x.args])


def mark_pointers(F):
    bad = set()
    base = set()
    ok_ops = {"+", "-", "==", "!=", "<u", "<=u", ">u", ">=u"}

    def visit(e, parent_op=None):
        if isinstance(e, Var):
            if parent_op is not None and parent_op not in ok_ops:
                bad.add(e.var)
            return
        if isinstance(e, Load):
            a = e.addr
            if isinstance(a, Var):
                base.add(a.var)
            elif isinstance(a, Bin) and a.op == "+" and isinstance(a.a, Var):
                base.add(a.a.var)
            visit(a, "+")
            return
        if isinstance(e, Bin):
            visit(e.a, e.op)
            visit(e.b, e.op if e.op in ("==", "!=") else ("rhs" if e.op in ("+",) else e.op))
            return
        if isinstance(e, (Cast,)):
            if isinstance(e.a, Var):
                bad.add(e.a.var)
            visit(e.a, "cast")
            return
        for c in e.children():
            visit(c, "other" if not isinstance(e, (Call, Select)) else None)

    for b in F.block_list():
        for s in b.stmts:
            if isinstance(s, Store):
                a = s.addr
                if isinstance(a, Var):
                    base.add(a.var)
                elif isinstance(a, Bin) and a.op == "+" and isinstance(a.a, Var):
                    base.add(a.a.var)
                visit(s.addr, "+")
                visit(s.val)
            else:
                for e in s.exprs():
                    visit(e)
            if isinstance(s, Assign) and isinstance(s.dst, Var):
                src = s.src
                if not (isinstance(src, (Var, Load, Call, Sym, Str)) or
                        (isinstance(src, Bin) and src.op in ("+", "-")) or isinstance(src, Const)):
                    bad.add(s.dst.var)
        if b.term is not None:
            for e in b.term.exprs():
                visit(e)
    # propagate through copies and pointer +/- integer
    links = []
    for b in F.block_list():
        for s in b.stmts:
            if isinstance(s, Assign) and isinstance(s.dst, Var):
                src = s.src
                if isinstance(src, Var):
                    links.append((s.dst.var, src.var))
                elif isinstance(src, Bin) and src.op in ("+", "-") and isinstance(src.a, Var) and \
                        not isinstance(src.b, Var):
                    links.append((s.dst.var, src.a.var))
    ptr = {v for v in base if v not in bad and v.ty == "i64"}
    changed = True
    while changed:
        changed = False
        for a, b2 in links:
            for x, y in ((a, b2), (b2, a)):
                if x in ptr and y not in ptr and y not in bad and y.ty == "i64":
                    ptr.add(y)
                    changed = True
    for v in F.vars:
        v.pointer = v in ptr


# ------------------------------------------------------------------ output
def signature(F):
    params = {}
    for v in F.vars:
        if v.is_param:
            params[v.param_index] = v
    used = set()
    defs, uses = an.collect(F)
    for k, v in params.items():
        if uses.get(v) or v.reg.startswith("stack"):
            used.add(k)
    n = max(used) + 1 if used else 0
    ps = []
    pr = Printer(F, None)
    for k in range(n):
        v = params.get(k)
        if v is None:
            ps.append("long a%d" % (k + 1))
        else:
            t = pr.vtype(v)
            ps.append("%s%s%s" % (t, "" if t.endswith("*") else " ", v.name))
    ret = SIGNED.get(F.ret_ty, "long")
    return "%s %s(%s)" % (ret, cname(F.name), ", ".join(ps) or "void"), n


def render(F, ctx, show_addr=False):
    ast = structure(F)
    pr = Printer(F, ctx)
    sig, n = signature(F)
    lines = [sig, "{"]
    defs, uses = an.collect(F)
    decl = []
    for v in F.vars:
        if v.is_input:
            if v.reg == "sp" or v.is_param:
                continue
            if uses.get(v):
                decl.append("    %s %s; // undefined (%s)" % (pr.vtype(v), v.name, v.reg))
            continue
        if v in defs:
            t = pr.vtype(v)
            decl.append("    %s%s%s; // %%%s" % (t, "" if t.endswith("*") else " ", v.name, v.reg))
    for L in getattr(F, "locals", []):
        t = SIGNED.get(L["ty"], "char")
        if L["count"] > 1:
            decl.append("    %s %s[%d]; // [sp-%#x]" % (t, L["name"], L["count"], -L["off"]))
        else:
            decl.append("    %s %s; // [sp-%#x]" % (t, L["name"], -L["off"]))
    lines.extend(decl)
    if decl:
        lines.append("")
    body = []
    pr.nodes(ast, 1, body)
    lines.extend(body)
    lines.append("}")
    if F.warnings:
        uniq = []
        for w in F.warnings:
            if w not in uniq:
                uniq.append(w)
        lines.insert(0, "// warnings: " + "; ".join(uniq[:8]) + (" ..." if len(uniq) > 8 else ""))
    if F.loop_bundles:
        lines.insert(0, "// note: function uses hardware-pipelined loops (rotating registers, AAU);"
                        " that part is shown at low level")
    return "\n".join(lines), sig


class Decompiler:
    def __init__(self, binary, objdump=None, dis_text=None):
        self.binary = binary
        try:
            self.elf = Elf(binary)
        except Exception:
            self.elf = None
        text = dis_text if dis_text is not None else od.run_objdump(binary, objdump)
        self.funcs = od.parse(text)
        self.res = Resolver(self.elf, self.funcs)
        self.ctx = Context(self.elf, self.res)

    def candidates(self, names=None):
        out = []
        for f in self.funcs:
            if names:
                if f.name in names:
                    out.append(f)
                continue
            if "@" in f.name or f.name in SKIP or not f.bundles:
                continue
            out.append(f)
        return out

    def decompile(self, names=None, header=True, dump_ir=False):
        funcs = self.candidates(names)
        summaries = {}
        # pass 1: parameter counts / void-ness
        for f in self.funcs:
            if not f.bundles or "@" in f.name:
                continue
            try:
                F = lift_function(f, self.res, {})
                if F is None:
                    continue
                optimize(F)
                summaries[f.name] = summarize(F)
            except Exception:
                pass
        results = []
        for f in funcs:
            try:
                jt = {}
                for _ in range(4):
                    F = lift_function(f, self.res, summaries, jt)
                    if F is None:
                        break
                    optimize(F)
                    resolve_exe_consts(F, self.ctx)
                    new = find_jump_tables(F, self.ctx)
                    if all(jt.get(k) == v for k, v in new.items()):
                        break
                    jt.update(new)
                if F is None:
                    continue
                to_switch(F)
                fix_outgoing_stack_args(F)
                layout_frame(F)
                an.simplify_all(F)
                an.propagate(F)
                an.dce(F)
                finalize_types(F, self.ctx, summaries)
                if dump_ir:
                    results.append((f.name, dump(F), None))
                    continue
                code, sig = render(F, self.ctx)
                results.append((f.name, code, sig))
            except Exception as ex:
                import traceback
                results.append((f.name, "// %s: decompilation failed: %s\n/*\n%s*/" %
                                (f.name, ex, traceback.format_exc()), None))
        out = []
        if header and not dump_ir:
            out.append(HEADER)
            defined = {name for name, _, _ in results}
            decls = self.ctx.declarations(defined)
            if decls:
                out.extend(decls)
                out.append("")
            sigs = [sig + ";" for _, _, sig in results if sig]
            if sigs:
                out.extend(sigs)
                out.append("")
        for name, code, sig in results:
            out.append(code)
            out.append("")
        return "\n".join(out)


def dump(F):
    lines = ["== %s" % F.name]
    for b in F.block_list():
        lines.append("%s: preds=%s succs=%s" % (b.name, [hex(p) for p in b.preds], [hex(s) for s in b.succs]))
        for s in b.stmts:
            lines.append("    %r" % (s,))
        lines.append("    -> %r" % (b.term.__class__.__name__,))
    return "\n".join(lines)
