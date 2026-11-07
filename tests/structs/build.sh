#!/bin/sh
# Build structs*_O0..O3.o and their listings.
#   LCC=/opt/mcst/lcc-1.31.05.e2k-v3.4c.linux-6.1 tests/structs/build.sh [structs structs2]
# When the toolchain is not installed under /opt/mcst, -B tells the lcc driver
# where its components are.
HERE=$(cd "$(dirname "$0")" && pwd)
M=${LCC:-/opt/mcst/lcc-1.31.05.e2k-v3.4c.linux-6.1}
H=$M/lcc-home/1.31.05/e2k-4c-linux.cross
OD=${OBJDUMP:-$HERE/../e2k-linux-objdump}
for B in ${*:-structs structs2}; do
    for O in 0 1 2 3; do
        "$M/bin/lcc" -B"$H/bin/" -B"$M/binutils/bin/e2k-linux-" -O$O -c "$HERE/$B.c" \
            -o "$HERE/${B}_O$O.o" || exit 1
        "$OD" -dr "$HERE/${B}_O$O.o" > "$HERE/${B}_O$O.dis"
    done
done
