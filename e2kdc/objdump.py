"""Run e2k objdump and parse its output into functions -> bundles -> ops."""
import os
import re
import shutil
import subprocess

RE_FUNC = re.compile(r"^([0-9a-f]+) <(.+)>:\s*$")
RE_BUNDLE = re.compile(r"^\s+([0-9a-f]+):\s*$")
RE_RELOC = re.compile(r"^\s+([0-9a-f]+): (R_\S+)\s+(.+?)\s*$")
RE_SECTION = re.compile(r"^Disassembly of section (\S+):")


class Op:
    """One elementary operation of a wide instruction (one text line)."""
    __slots__ = ("text", "mnem", "chan", "flags", "args", "guard", "extra")

    def __init__(self, text):
        self.text = text
        self.extra = {}
        self._parse(text)

    def _parse(self, text):
        t = text.strip()
        guard = None
        # trailing predicate:  "... ? %pred0"   "... ? ~%pred3"  "? %NOT_LOOP_END"
        m = re.search(r"\s\?\s(.+)$", t)
        if m:
            guard = m.group(1).strip()
            t = t[:m.start()].strip()
        self.guard = guard
        head, _, rest = t.partition(" ")
        parts = head.split(",")
        self.mnem = parts[0]
        self.chan = None
        self.flags = set()
        for p in parts[1:]:
            if p.isdigit():
                self.chan = int(p)
            else:
                self.flags.add(p)
        self.args = split_args(rest.strip())

    def __repr__(self):
        return self.text.strip()


def split_args(s):
    if not s:
        return []
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


class Bundle:
    __slots__ = ("addr", "ops", "relocs", "size", "func")

    def __init__(self, addr):
        self.addr = addr
        self.ops = []
        self.relocs = []      # (offset, type, target-string)
        self.size = 8

    def __repr__(self):
        return "%x: %s" % (self.addr, "; ".join(o.text.strip() for o in self.ops))


class Function:
    def __init__(self, name, addr, section):
        self.name, self.addr, self.section = name, addr, section
        self.bundles = []

    def __repr__(self):
        return "<Function %s @%#x, %d bundles>" % (self.name, self.addr, len(self.bundles))


def find_objdump(explicit=None):
    cands = [explicit, os.environ.get("E2K_OBJDUMP"), "e2k-linux-objdump"]
    for c in cands:
        if not c:
            continue
        p = shutil.which(c) or (c if os.path.isfile(c) else None)
        if p:
            return p
    raise FileNotFoundError("e2k objdump not found: pass --objdump or set E2K_OBJDUMP")


def run_objdump(binary, objdump=None):
    exe = find_objdump(objdump)
    res = subprocess.run([exe, "-dr", "--no-show-raw-insn", binary],
                         capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        raise RuntimeError("objdump failed: " + res.stderr)
    return res.stdout


def parse(text):
    funcs = []
    section = None
    func = None
    bundle = None
    for line in text.splitlines():
        if not line.strip():
            continue
        m = RE_SECTION.match(line)
        if m:
            section = m.group(1)
            continue
        m = RE_FUNC.match(line)
        if m:
            func = Function(m.group(2), int(m.group(1), 16), section)
            funcs.append(func)
            bundle = None
            continue
        if func is None:
            continue
        m = RE_RELOC.match(line)
        if m and line.startswith("\t"):
            off = int(m.group(1), 16)
            # attach to the bundle containing the offset
            tgt = None
            for b in reversed(func.bundles):
                if b.addr <= off:
                    tgt = b
                    break
            if tgt is not None:
                tgt.relocs.append((off, m.group(2), m.group(3)))
            continue
        m = RE_BUNDLE.match(line)
        if m:
            bundle = Bundle(int(m.group(1), 16))
            bundle.func = func
            func.bundles.append(bundle)
            continue
        if bundle is not None and line.startswith("  "):
            if line.strip().startswith("..."):
                continue
            bundle.ops.append(Op(line))
    for f in funcs:
        for i, b in enumerate(f.bundles):
            if i + 1 < len(f.bundles):
                b.size = f.bundles[i + 1].addr - b.addr
    return funcs
