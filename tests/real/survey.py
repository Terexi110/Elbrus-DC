#!/usr/bin/env python3
"""Regression on real programs: decompile e2k executables and count what is
left over.

    python3 tests/real/survey.py                  # utilities of the lcc sysroot
    python3 tests/real/survey.py bin1 bin2 ...    # any e2k ELF files
    python3 tests/real/survey.py -o out/ ...      # also keep the .c files

The default set is the glibc utilities in $LCC/fs/usr/bin (LCC defaults to
/opt/mcst/lcc-1.31.05.e2k-v3.4c.linux-6.1).  They are position-independent
executables, so they check %ip-relative addressing and calls through .got.

Per file: functions, functions that failed (exception or time budget),
functions still containing __e2k_* / __asm__, goto statements, gcc errors
(-fsyntax-only, warnings off) and seconds.  Nothing is compared with a
reference: the numbers are for watching trends between versions.
"""
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from e2kdc.decompiler import Decompiler   # noqa: E402

DEFAULT = ["getconf", "pcprofiledump", "pldd", "zdump", "getent", "gencat", "makedb", "sprof"]
FUNC = re.compile(r"^[A-Za-z_][^\n;]*?\b(\w+)\([^\n;]*\)\n\{\n(.*?)^\}", re.M | re.S)
GCC = ["gcc", "-c", "-w", "-fsyntax-only", "-Wno-int-conversion", "-Wno-incompatible-pointer-types",
       "-Wno-implicit-function-declaration"]


def gcc_errors(code):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.c")
        with open(p, "w") as f:
            f.write(code)
        try:
            r = subprocess.run(GCC + [p, "-o", os.devnull], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stderr.count("error:")


def survey(path, outdir=None, timeout=60, jobs=0):
    t = time.time()
    d = Decompiler(path, objdump=os.environ.get("E2K_OBJDUMP") or os.path.join(ROOT, "tests", "e2k-linux-objdump"))
    d.func_timeout = timeout
    d.jobs = jobs
    code = d.decompile(header=True)
    secs = time.time() - t
    if outdir:
        with open(os.path.join(outdir, os.path.basename(path) + ".c"), "w") as f:
            f.write(code)
    funcs = FUNC.findall(code)
    failed = re.findall(r"^// (\w+): decompilation failed", code, re.M)
    low = sum(1 for _, b in funcs if re.search(r"__e2k_\w+|__asm__", b))
    gotos = sum(len(re.findall(r"\bgoto\b", b)) for _, b in funcs)
    return {"file": os.path.basename(path), "funcs": len(funcs) + len(failed), "failed": len(failed),
            "low": low, "goto": gotos, "errors": gcc_errors(code), "secs": secs}


def main(argv):
    outdir = None
    jobs = 0                         # worker processes per file (0: one per CPU)
    while len(argv) >= 2 and argv[0] in ("-o", "-j"):
        if argv[0] == "-o":
            outdir = argv[1]
            os.makedirs(outdir, exist_ok=True)
        else:
            jobs = int(argv[1])
        argv = argv[2:]
    if not argv:
        lcc = os.environ.get("LCC", "/opt/mcst/lcc-1.31.05.e2k-v3.4c.linux-6.1")
        argv = [os.path.join(lcc, "fs", "usr", "bin", n) for n in DEFAULT]
    tot = {"funcs": 0, "failed": 0, "low": 0, "goto": 0, "errors": 0, "secs": 0.0}
    print("%-16s %6s %6s %6s %6s %7s %6s" % ("file", "funcs", "failed", "__e2k", "goto", "gccerr", "sec"))
    for p in argv:
        if not os.path.exists(p):
            print("%-16s missing" % os.path.basename(p))
            continue
        try:
            r = survey(p, outdir, jobs=jobs)
        except Exception as ex:          # the file as a whole: objdump, ELF
            print("%-16s CRASH %s" % (os.path.basename(p), ex))
            continue
        print("%-16s %6d %6d %6d %6d %7s %6.0f" % (r["file"], r["funcs"], r["failed"], r["low"], r["goto"],
                                                 "-" if r["errors"] is None else r["errors"], r["secs"]))
        for k in tot:
            tot[k] += r[k] or 0
    print("%-16s %6d %6d %6d %6d %7d %6.0f" % ("total", tot["funcs"], tot["failed"], tot["low"], tot["goto"],
                                             tot["errors"], tot["secs"]))


if __name__ == "__main__":
    main(sys.argv[1:])
