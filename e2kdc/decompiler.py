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
    if "@" in name and not name.startswith("@"):
        name = name.split("@")[0]          # stderr@GLIBC_2.2 (a symbol version)
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
        # executables: function by virtual address, whatever the section
        self.by_vaddr = {f.addr: f for f in funcs} if elf is not None and not elf.is_rel else {}
        self.current = None

    @staticmethod
    def clean_func_name(n):
        # printf@plt, free@plt.got, printf@secondary_plt, error@GLIBC_2.2
        if "@" in n and not n.startswith("@"):
            return n.split("@")[0]
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
        if f is None and self.elf is not None and not self.elf.is_rel:
            # executable: addresses are global, the target may be in another
            # section (calls from .text into the `printf@plt` stubs of .plt)
            g = self.by_vaddr.get(addr)
            f = self.clean_func_name(g.name) if g else None
            if f is None:
                sec = self.elf.section_of_addr(addr)
                if sec is not None and sec.flags & 4:     # SHF_EXECINSTR: unnamed code
                    f = "sub_%x" % addr
        if f:
            return ("func", f)
        return ("abs", addr)

    def lit_inplace(self, bundle, rel):
        """The value stored in the literal syllable a relocation patches (what
        objdump prints for it), or None."""
        if self.elf is None or not self.elf.is_rel:
            return None
        sec = self.elf.section(bundle.func.section)
        if sec is None:
            return None
        n = 8 if "64" in rel[1] else 4
        raw = self.elf.read(sec, rel[0], n)
        if not raw or len(raw) != n:
            return None
        if n == 8 and "LIT" in rel[1]:
            # a 64-bit literal takes two syllables, the high half first
            return int.from_bytes(raw[:4], "little") << 32 | int.from_bytes(raw[4:], "little")
        return int.from_bytes(raw, "little")

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
        # the in-place literal is the section-relative address (symbol value +
        # addend), not an extra offset: only a section-symbol relocation without
        # addend takes its offset from it
        e = Sym(name, add + (val if val and not add and name.startswith(".") else 0), kind)
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
        from .structs import Registry
        self.structs = Registry()
        self.global_structs = {}  # data symbol -> (Struct, element count or None)
        self.global_use = {}          # data symbol -> (types, constant offsets, indexed)
        self.defined = {}       # function name -> signature
        self.proto_log = None   # list: add_proto records its calls (worker processes)

    def add_proto(self, name, ret, args):
        """a call of name with these argument types (ret: the type of the
        first call's result); the calls of a file are seen in function order"""
        if self.proto_log is not None:
            self.proto_log.append((name, ret, args))
        if name not in self.protos:
            self.protos[name] = (ret, args)
        else:
            # call sites disagree on the count: a variadic function,
            # declared with the parameters every call passes
            ret, old = self.protos[name][:2]
            var = len(self.protos[name]) > 2 or len(old) != len(args)
            common = [a if a == b else "long" for a, b in zip(old, args)]
            self.protos[name] = (ret, common, True) if var else (ret, common)

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

    def symbol_size(self, name):
        for s in (self.elf.symbols if self.elf else []):
            if s.name == name and s.size:
                return s.size
        return None

    def resolve_local(self, sym, F):
        """a stack local folded into a structure object -> (object, offset)"""
        al = getattr(F, "local_alias", {}).get(sym.name) if sym.kind == "local" else None
        if al:
            return Sym(al[0], al[1] + sym.off, "local")
        return sym

    def object_struct(self, sym, F):
        """(name, Struct, element count, offset) of a global / stack object of
        structure type containing sym, else None"""
        if sym.kind == "local":
            sym = self.resolve_local(sym, F)
            L = self.local_info(sym, F)
            if L is not None and L.get("struct") is not None:
                return L["name"], L["struct"], None, sym.off
            return None
        n = self.normalize(sym)
        g = self.global_structs.get(n.name)
        if g is None:
            return None
        self.note_global(n.name, "struct")
        return cname(n.name), g[0], g[1], n.off - 0

    def local_info(self, sym, F):
        sym = self.resolve_local(sym, F)
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
        use = self.global_use.get(sym.name)
        size = self.symbol_size(sym.name)
        esz = BITS[ty] // 8 if ty in BITS else 8
        aggregate = (size is not None and size > esz) or \
            (use is not None and (use[2] or any(o != 0 for o in use[1])))
        if sym.off == 0 and not aggregate:
            self.note_global(sym.name, ty)
            return n
        if use is not None and use[0] == {ty} and sym.off % esz == 0 and sym.off >= 0:
            # an array of one element type: gtab[6]
            self.note_global(sym.name, ty, array=True)
            return "%s[%d]" % (n, sym.off // esz)
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
            x = self.resolve_local(x, F)
            L = self.local_info(x, F)
            if L and L.get("struct") is not None:
                return ("&" + L["name"], 14, "struct %s *" % __import__("e2kdc.structs", fromlist=["rep"]).rep(L["struct"]).name) \
                    if x.off == 0 else ("(char *)&%s + %d" % (L["name"], x.off), 12, "char *")
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
        from .structs import rep
        out = list(self.structs.definitions())
        for name, (S, count) in sorted(self.global_structs.items()):
            if name in self.globals:
                out.append("extern struct %s %s%s;" % (rep(S).name, cname(name),
                                                      "[%d]" % count if count else ""))
        for name, g in sorted(self.globals.items()):
            if name in self.global_structs:
                continue
            n = cname(name)
            tys = g["types"]
            if len(tys) == 1 and not g["array"]:
                out.append("extern %s %s;" % (SIGNED.get(next(iter(tys)), "long"), n))
            elif len(tys) == 1 and g["array"]:
                t = next(iter(tys))
                size, esz = self.symbol_size(name), BITS.get(t, 64) // 8
                out.append("extern %s %s[%s];" % (SIGNED.get(t, "long"), n,
                                                  size // esz if size and size % esz == 0 else ""))
            else:
                out.append("extern char %s[];" % n)
        for name, pr in sorted(self.protos.items()):
            ret, args = pr[0], pr[1]
            variadic = len(pr) > 2
            if name in funcs_defined:
                continue
            if name in VARIADIC:
                out.append(VARIADIC[name])
                continue
            from . import libc
            if libc.lookup(name):
                out.append(libc.lookup(name)[0])
                continue
            if variadic:
                out.append("%s %s(%s);" % (ret, cname(name), ", ".join(args + ["..."]) if args else ""))
            else:
                out.append("%s %s(%s);" % (ret, cname(name), ", ".join(args) or "void"))
        return out


# =================================================================== per function
def lift_function(func, resolver, summaries, jtables=None, named_calls=None):
    F = cfgmod.build(func, resolver, jtables)
    if not F.blocks:
        return None
    # calls through a .got slot named by an earlier round (PIC): known before
    # the arguments are assigned, so the prototype gives their number
    for blk, st in F.calls:
        t = (named_calls or {}).get(getattr(st, "addr", None))
        if t is not None and not isinstance(st.src.target, Sym):
            st.src.target = t
    an.decide_void(F)
    if summaries and summaries.get(func.name, {}).get("void") and F.returns_value and \
            summaries.get(func.name, {}).get("callers_void"):
        for b in F.block_list():
            if isinstance(b.term, Return):
                b.term = Return(None)
        F.returns_value = False
    nparams_hint = summaries.get(func.name, {}).get("nparams") if summaries else None
    an.assign_call_args(F, summaries or {}, fparams=nparams_hint)
    from . import libc
    for blk, st in F.calls:
        tn = st.src.target.name if isinstance(st.src.target, Sym) else None
        if tn and summaries.get(tn, {}).get("void") and summaries[tn].get("callers_void"):
            st.src.ret_used = False
        elif tn and (summaries.get(tn, {}).get("void") or
                     (tn not in summaries and libc.lookup(tn) and libc.lookup(tn)[1] == "void")):
            # a void library function (error, free) defines no result: the
            # register keeps no value of it, and C must not read one
            st.src.ret_used = False
            for i, x in enumerate(blk.stmts):
                if x is st:
                    es = ExprStmt(st.src, st.guard, getattr(st, "addr", None))
                    es.is_call = True
                    blk.stmts[i] = es
                    break
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
    if an.dce_live(F):          # dead copy cycles (rotating registers)
        an.propagate(F)
        an.dce(F)
    an.const_intr_args(F)       # constant field specs -> plain shifts/masks
    an.simplify_all(F)
    from .structure import fix_branches, prune_unreachable, merge_chains
    fix_branches(F)             # constant branches (? %MLOCK) -> one-block kernels
    prune_unreachable(F)
    merge_chains(F)
    from .structure import sink_guarded, sink_guarded_far
    if sink_guarded(F) | sink_guarded_far(F):   # if (p) S; ... branch p  ->  S inside the p-arm
        an.propagate(F)
        an.dce(F)
    from .depipe import collapse_pipelines, rebase_all
    if collapse_pipelines(F):   # software-pipelined loops -> one iteration per trip
        an.simplify_all(F)
        an.dce_live(F)
        an.propagate(F)
        an.dce(F)
        rebase_all(F)
        an.simplify_all(F)
        an.propagate(F)
        an.dce(F)
    an.const_intr_args(F, all_vars=True)   # v = 0 reaching a kernel through a merged web
    an.simplify_all(F)
    an.dce_live(F)
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
    from .types import returns_32
    return {"nparams": n, "void": not F.returns_value, "ret32": returns_32(F)}


def note_callers(F, callers):
    """callee name -> True if some call in F uses the returned value"""
    for b in F.block_list():
        for st in b.stmts:
            top = st.e if isinstance(st, ExprStmt) else None
            for e in st.exprs():
                for x in e.walk():
                    if isinstance(x, Call) and isinstance(x.target, Sym) and x.target.kind == "func":
                        used = x is not top and getattr(x, "ret_used", True)
                        callers[x.target.name] = callers.get(x.target.name, False) or used
        for e in (b.term.exprs() if b.term is not None else []):
            for x in e.walk():
                if isinstance(x, Call) and isinstance(x.target, Sym) and x.target.kind == "func":
                    callers[x.target.name] = True


def has_uses(F, v):
    defs, uses = an.collect(F)
    return bool(uses.get(v))


# ------------------------------------------------------------------ finalize passes
R_E2K_64_ABS, R_E2K_64_JMP_SLOT, R_E2K_64_COPY, R_E2K_64_RELATIVE = 50, 63, 64, 65


def dyn_relocs(elf):
    """virtual address -> dynamic relocation (executables, shared objects)"""
    d = getattr(elf, "_dynrel", None)
    if d is None:
        d = {}
        if not elf.is_rel:
            for lst in elf.relocs.values():
                for r in lst:
                    d.setdefault(r.offset, r)
        elf._dynrel = d
    return d


def got_value(elf, addr):
    """What the dynamic linker puts at addr (a .got slot): Sym of the imported
    function / object, a link-time constant for R_E2K_64_RELATIVE, else None."""
    r = dyn_relocs(elf).get(addr)
    if r is None:
        return None
    if r.type == R_E2K_64_RELATIVE:
        return Const(r.addend, "i64")
    if r.type in (R_E2K_64_ABS, R_E2K_64_JMP_SLOT) and r.sym is not None and r.sym.name:
        if r.sym.shndx and r.sym.value:          # defined here: its address in this file
            return Const(r.sym.value + r.addend, "i64")
        kind = "func" if r.type == R_E2K_64_JMP_SLOT or r.sym.type == STT_FUNC else "data"
        return Sym(r.sym.name, r.addend, kind)
    return None


def resolve_exe_consts(F, ctx):
    elf = ctx.elf
    if not elf or elf.is_rel:
        return

    def slot(e):
        # PIC: `ldd [%ip + k]` reads a .got slot -> the address the loader put there
        if isinstance(e, Load) and e.ty == "i64":
            a = e.addr
            if isinstance(a, Const):
                return got_value(elf, a.value)
            if isinstance(a, Sym) and a.kind == "data":
                sec = elf.section(a.name)
                base = sec.addr if sec is not None else next(
                    (x.value for x in elf.symbols if x.name == a.name and x.shndx), None)
                if base is not None:
                    return got_value(elf, base + a.off)
        return None

    def fix(e):
        g = slot(e)
        if g is not None:
            return fix(g)
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
            if not isinstance(call, Call) or (getattr(call, "known_arity", False) and
                                              getattr(call, "nfixed", 0) < 8):
                # a callee with fewer than 8 parameters takes nothing on the stack:
                # stores near %sp before it are the caller's own locals (with 8
                # register parameters a summary cannot tell whether more follow)
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
    # library prototypes, pointer-like variables, then 64-bit variables holding 32-bit values
    from .types import apply_libc
    hints = apply_libc(F, ctx)
    for v, h in hints.items():
        v.pointee_hint = h
    an.dce(F)
    mark_pointers(F, hints)
    from .types import (demote_ints, ret_kind, signedness, expr_unsigned, narrow_params, func_pointers,
                        float_vars, fold_float_mem, fold_float_store)
    float_vars(F)
    for b in F.block_list():
        for st in b.stmts:
            st.map(fold_float_mem)
            fold_float_store(st)
        if b.term is not None:
            b.term.map(fold_float_mem)
    demote_ints(F)
    narrow_params(F)
    signedness(F)
    func_pointers(F)
    # return type
    rets = [b.term for b in F.block_list() if isinstance(b.term, Return)]
    vals = [r.value for r in rets if r.value is not None]
    from .ir import BitCast as _BC

    def fbits(v):
        while isinstance(v, Cast) and BITS.get(v.ty, 64) >= BITS.get(v.a.ty, 64) and not v.signed:
            v = v.a
        return v.a if isinstance(v, _BC) and is_float(v.a.ty) else None
    if vals and all(fbits(v) is not None for v in vals) and len({fbits(v).ty for v in vals}) == 1:
        for r in rets:
            if r.value is not None:
                r.value = fbits(r.value)        # a float returned in %r0
        vals = [r.value for r in rets if r.value is not None]
    rty = "v"
    F.ret_unsigned = False
    rk = ret_kind(vals) if vals else None
    if vals and rk is not None and not any(is_float(v.ty) for v in vals) and \
            not all(v.ty in ("i32",) for v in vals) and \
            not all(isinstance(v, Cast) and v.a.ty in ("i8", "i16") and v.ty == "i64" for v in vals):
        rty = "i32"
        F.ret_unsigned = rk == frozenset("u")
        for r in rets:
            if r.value is not None and r.value.ty != "i32":
                r.value = simplify(Cast(r.value, "i32", signed=True))
        vals = []
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
    rvals = [r.value for r in rets if r.value is not None]
    if rty in ("i32", "i64") and rvals and all(expr_unsigned(v) or isinstance(v, Const) for v in rvals) and \
            not all(isinstance(v, Const) for v in rvals):
        F.ret_unsigned = True
    # prototypes for called functions
    def arg_ctype(a):
        if isinstance(a, (Str,)) or (isinstance(a, Sym) and a.kind in ("data", "local")) or \
                (isinstance(a, Var) and getattr(a.var, "pointer", False)):
            return "void *"
        return SIGNED.get(a.ty, "long") if a.ty != "b" else "int"
    for b in F.block_list():
        exprs = [e for s in b.stmts for e in s.exprs()] + (list(b.term.exprs()) if b.term is not None else [])
        for e in exprs:
            for x in e.walk():
                if isinstance(x, Call) and isinstance(x.target, Sym) and x.target.kind == "func":
                    name = x.target.name
                    args = [arg_ctype(a) for a in x.args]
                    ret = SIGNED.get(x.ty, "long") if x.ret_used else "long"
                    if summaries.get(name, {}).get("void"):
                        ret = "void"
                    ctx.add_proto(name, ret, args)


def mark_pointers(F, hints=()):
    bad = set()
    base = set(v for v in hints if v.ty == "i64")
    ok_ops = {"+", "-", "==", "!=", "<u", "<=u", ">u", ">=u", "<s", "<=s", ">s", ">=s"}

    casted = set()               # (int)p: a pointer difference truncated to int

    def visit(e, parent_op=None):
        if isinstance(e, Var):
            if parent_op in ("cast", "rhs"):
                casted.add(e.var)    # weak: (int)p, q + p - r (rebased pointers)
            elif parent_op is not None and parent_op not in ok_ops:
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
        if isinstance(e, Bin) and e.op == "&" and isinstance(e.b, Const) and e.b.value < 4096:
            visit(e.a, "+")          # alignment test (p & 7): still a pointer
            return
        if isinstance(e, Bin):
            visit(e.a, e.op)
            visit(e.b, e.op if e.op in ("==", "!=") else ("rhs" if e.op in ("+",) else e.op))
            return
        if isinstance(e, (Cast,)):
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
                elif isinstance(src, Bin) and src.op == "+" and isinstance(src.b, Var) and \
                        not isinstance(src.a, Var):
                    links.append((s.dst.var, src.b.var))      # index*4 + p
    # (int)p alone does not make p an integer: pointer differences are truncated
    ptr = {v for v in base if v not in bad and v.ty == "i64"}
    changed = True
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
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
    live = an.live_at_entry(F)
    for k, v in params.items():
        # a parameter is used if its incoming value can be read (r7 merely written
        # before use under the same predicate as the early return is not a parameter)
        if (uses.get(v) and v in live) or v.reg.startswith("stack"):
            used.add(k)
    n = max(used) + 1 if used else 0
    ps = []
    pr = Printer(F, None)
    for k in range(n):
        v = params.get(k)
        if v is None:
            ps.append("long a%d" % (k + 1))
        else:
            ps.append(pr.decl(v))
    ret = SIGNED.get(F.ret_ty, "long")
    if getattr(F, "ret_unsigned", False) and F.ret_ty in UNSIGNED:
        ret = UNSIGNED[F.ret_ty]
    if getattr(F, "ret_pointer", None):
        ret = F.ret_pointer
    return "%s %s(%s)" % (ret, cname(F.name), ", ".join(ps) or "void"), n


def nullable_ptrs(ast):
    """pointer variables compared with 0 somewhere in the function"""
    from .devec import _roots, _exprs_of
    out = set()
    copies = []
    for h in _roots(ast):
        if isinstance(h, Assign) and isinstance(h.dst, Var) and isinstance(h.src, Var):
            copies.append((h.dst.var, h.src.var))
        for e in _exprs_of(h):
            for x in e.walk():
                if isinstance(x, Bin) and x.op in ("==", "!=") and isinstance(x.b, Const) and \
                        x.b.value == 0 and isinstance(x.a, Var) and getattr(x.a.var, "pointer", False):
                    out.add(x.a.var)
    changed = True                       # copies of a NULL-tested pointer may be NULL too
    _rounds = 0
    while changed and _rounds < 1000:   # a rewrite that undoes itself must not hang
        _rounds += 1
        changed = False
        for d, s in copies:
            for a, b in ((d, s), (s, d)):
                if b in out and a not in out and getattr(a, "pointer", False):
                    out.add(a)
                    changed = True
    return out


def prepare(F, ctx):
    """Structured AST of F with the per-function type facts (pointees, structs)."""
    ast = structure(F)
    from .types import pointees, fold_float_mem, fold_float_store, apply_hints
    from .devec import _roots, _map_root, _exprs_of
    for h in _roots(ast):
        _map_root(h, fold_float_mem)
        fold_float_store(h)
    from .simplify import narrow_rotates
    narrow_rotates(_roots(ast), _map_root)
    from .structs import rescale_offsets, infer as infer_structs
    rescale_offsets(ast, F)
    found = infer_structs(_roots(ast), _exprs_of)
    apply_hints(F, pointees(ast))
    return ast, found


def render(F, ctx, show_addr=False, prepared=None):
    ast, found = prepared if prepared is not None else prepare(F, ctx)
    if prepared is None:
        ctx.structs.register(found)
    pr = Printer(F, ctx)
    pr.nullable = nullable_ptrs(ast)
    sig, n = signature(F)
    # registers that looked like parameters beyond the last real one are locals
    demoted = False
    for v in F.vars:
        if v.is_param and v.param_index is not None and v.param_index >= n and not v.reg.startswith("stack"):
            v.is_param = False
            v.is_input = v in an.live_at_entry(F)
            demoted = True
    if demoted:
        an.name_vars(F)
        sig, n = signature(F)
    lines = [sig, "{"]
    defs, uses = an.collect(F)
    from .loops import count_vars
    live = count_vars(ast)
    decl = []
    for v in F.vars:
        if v.is_input:
            if v.reg == "sp" or v.is_param:
                continue
            if uses.get(v):
                decl.append("    %s %s; // undefined (%s)" % (pr.vtype(v), v.name, v.reg))
            continue
        if (v in defs or getattr(v, "ast_def", False)) and (live["defs"].get(v) or live["uses"].get(v)):
            decl.append("    %s; // %%%s" % (pr.decl(v), v.reg))
    body = []
    pr.nodes(ast, 1, body)
    text = "\n".join(body)
    import re as _re
    for L in getattr(F, "locals", []):
        if not _re.search(r"\b%s\b" % _re.escape(L["name"]), text):
            continue                     # e.g. the spill area of variadic arguments
        if L.get("struct") is not None:
            from .structs import rep
            decl.append("    struct %s %s; // [sp-%#x]" % (rep(L["struct"]).name, L["name"], -L["off"]))
            continue
        t = SIGNED.get(L["ty"], "char")
        if L["count"] > 1:
            decl.append("    %s %s[%d]; // [sp-%#x]" % (t, L["name"], L["count"], -L["off"]))
        else:
            decl.append("    %s %s; // [sp-%#x]" % (t, L["name"], -L["off"]))
    lines.extend(decl)
    if decl:
        lines.append("")
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


class FunctionTimeout(Exception):
    pass


class time_budget:
    """with time_budget(sec): ... raises FunctionTimeout when the block runs
    longer (SIGALRM; a no-op outside the main thread or without a budget)"""

    def __init__(self, sec):
        self.sec = sec
        self.armed = False

    def __enter__(self):
        import signal
        import threading
        if not self.sec or threading.current_thread() is not threading.main_thread() or \
                not hasattr(signal, "setitimer"):
            return self

        def fire(signum, frame):
            raise FunctionTimeout("time budget of %gs exceeded" % self.sec)
        self.old = signal.signal(signal.SIGALRM, fire)
        signal.setitimer(signal.ITIMER_REAL, self.sec)
        self.armed = True
        return self

    def __exit__(self, *exc):
        if self.armed:
            import signal
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.old)
        return False


def failed_stub(name, ex):
    import traceback
    if isinstance(ex, FunctionTimeout):
        return "// %s: decompilation failed: %s\n" % (name, ex)
    return "// %s: decompilation failed: %s\n/*\n%s*/" % (name, ex, traceback.format_exc())


class Decompiler:
    # seconds per function and stage (lifting+analysis, printing); 0: no limit
    func_timeout = 60
    # worker processes for the per-function passes (0: one per CPU); the
    # output does not depend on it
    jobs = 1

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

    def _analyze(self, f, summaries, dump_ir):
        """lift, analyse and structure one function: (name, F, prepared) or None"""
        jt, named = {}, {}
        for _ in range(4):
            F = lift_function(f, self.res, summaries, jt, named)
            if F is None:
                break
            optimize(F)
            resolve_exe_consts(F, self.ctx)
            new = find_jump_tables(F, self.ctx)
            got = {st.addr: st.src.target for blk, st in F.calls
                   if isinstance(st.src, Call) and isinstance(st.src.target, Sym) and
                   st.src.target.kind == "func" and getattr(st, "addr", None) not in named}
            got = {a: t for a, t in got.items() if a is not None}
            if all(jt.get(k) == v for k, v in new.items()) and not got:
                break
            jt.update(new)
            named.update(got)
        if F is None:
            return None
        to_switch(F)
        fix_outgoing_stack_args(F)
        layout_frame(F)
        an.simplify_all(F)
        an.propagate(F)
        an.dce(F)
        from .structure import drop_known_guards
        if drop_known_guards(F):
            an.simplify_all(F)
            an.propagate(F)
            an.dce(F)
        if an.split_webs(F):     # registers reused for unrelated values
            an.propagate(F)
            an.dce(F)
            an.name_vars(F)
        finalize_types(F, self.ctx, summaries)
        an.refresh_inputs(F)
        if dump_ir:
            return (f.name, dump(F), None)
        return (f.name, F, prepare(F, self.ctx))

    def _fallback_structs(self, live):
        """link_types failed: register each function's own structures, and drop
        the structure types of a function whose layouts cannot be registered"""
        from .structs import rep
        self.ctx.global_structs.clear()
        for name, F, (ast, found) in live:
            try:
                self.ctx.structs.register(found)
            except Exception:
                for v in F.vars:
                    v.struct = None
                for L in getattr(F, "locals", []):
                    if L.get("struct") is not None:
                        L["struct"] = None
        for name, F, (ast, found) in live:
            for v in F.vars:
                if getattr(v, "struct", None) is not None and not rep(v.struct).name:
                    v.struct = None

    def _first_pass(self, f):
        """(summary or None, [(callee, its result is used)]) of one function"""
        summ, calls = None, {}
        try:
            with time_budget(self.func_timeout):
                F = lift_function(f, self.res, {})
                if F is not None:
                    optimize(F)
                    summ = summarize(F)
                    note_callers(F, calls)
        except Exception:
            pass
        return summ, list(calls.items())

    def _second_pass(self, f, summaries, dump_ir):
        try:
            with time_budget(self.func_timeout):
                return self._analyze(f, summaries, dump_ir)
        except Exception as ex:
            return (f.name, failed_stub(f.name, ex), None)

    def decompile(self, names=None, header=True, dump_ir=False):
        funcs = self.candidates(names)
        summaries = {}
        callers = {}             # callee -> is its result used by some caller
        # pass 1: parameter counts / void-ness
        first = [f for f in self.funcs if f.bundles and "@" not in f.name]
        from .parallel import run as prun
        for f, (summ, calls) in zip(first, prun(self, "_first_pass", first, ())):
            if summ is not None:
                summaries[f.name] = summ
            for name, used in calls:
                callers[name] = callers.get(name, False) or used
        from . import types as _ty
        _ty.RET32.clear()
        _ty.RET32.update({k: v.get("ret32") for k, v in summaries.items()})
        # a function whose result no caller in this file reads is void
        for name, used in callers.items():
            if name in summaries and not used:
                summaries[name]["void"] = True
                summaries[name]["callers_void"] = True
        results = [r for r in prun(self, "_second_pass", funcs, (summaries, dump_ir), protos=True)
                   if r is not None]
        notes = []
        if not dump_ir:
            # types across functions: callee parameters, nested structures,
            # stack and global structures; then print.  A failure there costs
            # the cross-function types, not the output
            from .interproc import link_types
            live = [(n, F, p) for n, F, p in results if p is not None]
            try:
                with time_budget(2 * self.func_timeout):
                    link_types(live, self.ctx)
            except Exception as ex:
                notes.append("/* types across functions skipped: %s */" % ex)
                self._fallback_structs(live)
            done = []
            for name, F, p in results:
                if p is None:
                    done.append((name, F, None))
                    continue
                try:
                    with time_budget(self.func_timeout):
                        code, sig = render(F, self.ctx, prepared=p)
                    done.append((name, code, sig))
                except Exception as ex:
                    done.append((name, failed_stub(name, ex), None))
            results = done
        out = []
        if header and not dump_ir:
            out.append(HEADER)
            out.extend(notes)
            defined = {name for name, _, _ in results}
            try:
                decls = self.ctx.declarations(defined)
            except Exception as ex:
                decls = ["/* declarations failed: %s */" % ex]
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
