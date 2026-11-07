#!/bin/sh
# usage: ./dc.sh <binary.o|exe> [options]   (run from the Elbrus-DC folder)
HERE=$(cd "$(dirname "$0")" && pwd)
export E2K_OBJDUMP="${E2K_OBJDUMP:-$HERE/tests/e2k-linux-objdump}"
PYTHONPATH="$HERE" exec python3 -m e2kdc "$@"
