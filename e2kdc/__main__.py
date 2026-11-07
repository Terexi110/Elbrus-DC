"""e2kdc - decompiler of Elbrus (e2k) binaries into pseudo-C.

usage: python3 -m e2kdc [options] BINARY
"""
import argparse
import sys

from .decompiler import Decompiler


def main(argv=None):
    ap = argparse.ArgumentParser(prog="e2kdc", description="Elbrus (e2k) to pseudo-C decompiler")
    ap.add_argument("binary", help="e2k ELF file (.o, executable, .so)")
    ap.add_argument("-f", "--func", action="append", help="decompile only this function (repeatable)")
    ap.add_argument("--objdump", help="path to e2k-linux-objdump (or set E2K_OBJDUMP)")
    ap.add_argument("--dis", help="use an existing 'objdump -dr' listing instead of running objdump")
    ap.add_argument("--ir", action="store_true", help="dump intermediate representation")
    ap.add_argument("--no-header", action="store_true", help="omit helper macros and declarations")
    ap.add_argument("-o", "--output", help="write result to file")
    ap.add_argument("--timeout", type=float, default=60,
                    help="seconds per function and stage before it is given up (0: no limit)")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="worker processes for the per-function passes (default 0: one per CPU; "
                         "1: none); the output is the same")
    a = ap.parse_args(argv)
    text = None
    if a.dis:
        with open(a.dis, encoding="utf-8", errors="replace") as f:
            text = f.read()
    d = Decompiler(a.binary, objdump=a.objdump, dis_text=text)
    d.func_timeout = a.timeout
    d.jobs = a.jobs
    res = d.decompile(names=a.func, header=not a.no_header, dump_ir=a.ir)
    if a.output:
        with open(a.output, "w") as f:
            f.write(res)
    else:
        sys.stdout.write(res)


if __name__ == "__main__":
    main()
