#!/usr/bin/env python3
"""T7: broadcast local_load must select the l1bmp (replicated) mode.

Every lane and warp basis is zero, so all 64 PEs read the same 4 f32. The pass
must record `mode = 0` (Broadcast) and the conversion must produce
`llvm.mncore.l1load.v4f32`.

Usage: PYTHONPATH=python python run_t7.py t7_broadcast.mlir
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
    if '"mncore.vec.load"' not in text or "mode = 0" not in text:
        print("T7 FAIL: Broadcast mode was not selected on mncore.vec.load",
              file=sys.stderr)
        return 1

    pm2 = ir.pass_manager(ctx)
    pm2.enable_debug()
    mncore.add_convert_mncore_to_llvm(pm2)
    pm2.run(mod, "llir")
    llir = str(mod)
    if "llvm.mncore.l1load.v4f32" not in llir:
        print("T7 FAIL: no llvm.mncore.l1load.v4f32 produced", file=sys.stderr)
        return 1
    print("T7 OK: broadcast load -> l1bmp (mode 0) -> "
          "llvm.mncore.l1load.v4f32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
