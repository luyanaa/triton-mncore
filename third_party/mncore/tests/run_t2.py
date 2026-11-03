#!/usr/bin/env python3
"""T2: lower a minimal TTGIR elementwise kernel to the mncore dialect.

Usage: PYTHONPATH=python python run_t1.py t2_broadcast.mlir
"""

import pathlib
import sys

from triton._C.libtriton import ir
from triton._C.libtriton import mncore


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = str(pathlib.Path(sys.argv[1]).resolve())

    ctx = ir.context()
    ir.load_dialects(ctx)
    mncore.load_dialects(ctx)
    mod = ir.parse_mlir_module(path, ctx)

    pm = ir.pass_manager(ctx)
    pm.enable_debug()
    mncore.add_tritongpu_to_mncore(pm)
    pm.run(mod, "mnir")
    text = str(mod)
    print(text)
    if '"mncore.vec.load"' not in text:
        print("T2 FAIL: no mncore.vec.load produced", file=sys.stderr)
        return 1

    pm2 = ir.pass_manager(ctx)
    pm2.enable_debug()
    mncore.add_convert_mncore_to_llvm(pm2)
    pm2.run(mod, "llir")
    llir = str(mod)
    if "llvm.call_intrinsic" not in llir or "mncore" not in llir:
        print("T2 FAIL: mncore -> LLVM conversion did not run", file=sys.stderr)
        return 1
    print("T2 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
