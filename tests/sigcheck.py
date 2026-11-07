#!/usr/bin/env python3
"""Compare decompiled function signatures with the C sources of the tests.

    python3 tests/sigcheck.py            # all test objects
    python3 tests/sigcheck.py -v         # list every mismatch

Each parameter and the return type count as one slot.  A slot is "exact" when
the decompiled type equals the source type after normalisation (const dropped,
`unsigned` = `unsigned int`, any struct pointer = `struct *`), and "pointer"
when both are pointers but the pointee differs.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from e2kdc.decompiler import Decompiler   # noqa: E402

SETS = [("t1.c", ["t1_O0.o", "t1_O1.o", "t1_O2.o"]),
        ("t2.c", ["t2_O0.o", "t2_O1.o", "t2_O2.o"]),
        ("arch/arch.c", ["arch/arch_O1.o", "arch/arch_O3.o"]),
        ("loops/loops.c", ["loops/loops_O%d.o" % i for i in range(4)]),
        ("structs/structs.c", ["structs/structs_O%d.o" % i for i in range(4)]),
        ("structs/structs2.c", ["structs/structs2_O%d.o" % i for i in range(4)])]


def norm(t):
    t = " ".join(t.replace("const ", "").replace("*", " * ").split())
    if re.search(r"\(\s*\*", t):
        return "fptr"
    stars = t.count("*")
    base = t.replace("*", "").strip()
    base = {"unsigned": "unsigned int", "signed": "int", "long int": "long",
            "unsigned long int": "unsigned long", "_Bool": "int"}.get(base, base)
    if base.startswith("struct"):
        base = "struct"
    return base + " *" * stars


def split_params(s):
    s = s.strip()
    if s in ("", "void"):
        return []
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    res = []
    for p in out:
        p = p.strip()
        m = re.match(r"^(.*\(\s*\*\s*)(\w+)(\s*\).*)$", p)
        if m:
            res.append("fptr")
            continue
        m = re.match(r"^(.*?)(\w+)$", p)
        res.append(norm(m.group(1)) if m and m.group(1).strip() else norm(p))
    return res


SIG = re.compile(r"^([A-Za-z_][\w \*]*?)\s*\b(\w+)\s*\(([^;{}]*)\)\s*(\{|$)", re.M)


def source_sigs(path):
    text = open(path).read()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    sigs = {}
    for m in SIG.finditer(text):
        ret, name, params = m.group(1), m.group(2), m.group(3)
        if ret.strip() in ("return", "else", "if", "while", "for", "switch") or name in ("if", "while", "for", "switch"):
            continue
        if m.group(4) == "" and not re.match(r"\s*\{", text[m.end():]):
            continue
        ret = re.sub(r"\b(static|inline|extern|NOINLINE)\b", "", ret)
        sigs[name] = (norm(ret), split_params(params))
    return sigs


def dc_sigs(code):
    sigs = {}
    for m in re.finditer(r"^([A-Za-z_][\w \*]*?)\s*\b(\w+)\(([^;{}]*)\)\n\{", code, re.M):
        sigs[m.group(2)] = (norm(m.group(1)), split_params(m.group(3)))
    return sigs


def main():
    verbose = "-v" in sys.argv
    tot = ex = ptr = 0
    for src, objs in SETS:
        ss = source_sigs(os.path.join(HERE, src))
        for o in objs:
            path = os.path.join(HERE, o)
            if not os.path.exists(path):
                continue
            d = Decompiler(path, objdump=os.path.join(HERE, "e2k-linux-objdump"))
            ds = dc_sigs(d.decompile(None, header=False))
            oe = ot = 0
            for name, (sret, sps) in ss.items():
                if name not in ds or name == "main":
                    continue
                dret, dps = ds[name]
                slots = [("ret", sret, dret)] + [("a%d" % (i + 1), s, dps[i] if i < len(dps) else "-")
                                                 for i, s in enumerate(sps)]
                for slot, s, dty in slots:
                    tot += 1
                    ot += 1
                    if s == dty:
                        ex += 1
                        oe += 1
                    elif s.endswith("*") and dty.endswith("*"):
                        ptr += 1
                        if verbose:
                            print("  %-14s %-10s %-4s src %-16s dc %s" % (o, name, slot, s, dty))
                    elif verbose:
                        print("  %-14s %-10s %-4s src %-16s dc %s" % (o, name, slot, s, dty))
            print("%-16s %3d/%3d exact" % (o, oe, ot))
    print("total: %d slots, exact %d (%.0f%%), pointer with other pointee %d" %
          (tot, ex, 100.0 * ex / max(tot, 1), ptr))


if __name__ == "__main__":
    main()
