#!/usr/bin/env python3
"""T6: MAB4-aware local_load must select the l1bmm4 (4x4) mode.

Lane bits 2..3 are replicated and bit 4 (plus the warp bit) selects the 4-MAB
group, so the distribution is MAB4 (SDM 3.6.8.14). The pass must record
`mode = 2` and the conversion must produce `llvm.mncore.l1load.v4f32`.

Usage: PYTHONPATH=python python run_t6.py t6_mab4.mlir
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
    if '"mncore.vec.load"' not in text or "mode = 2" not in text:
        print("T6 FAIL: MAB4 mode was not selected on mncore.vec.load",
              file=sys.stderr)
        return 1

    pm2 = ir.pass_manager(ctx)
    pm2.enable_debug()
    mncore.add_convert_mncore_to_llvm(pm2)
    pm2.run(mod, "llir")
    llir = str(mod)
    if "llvm.mncore.l1load.v4f32" not in llir:
        print("T6 FAIL: no llvm.mncore.l1load.v4f32 produced", file=sys.stderr)
        return 1
    print("T6 OK: MAB4 load -> l1bmm4 (mode 2) -> llvm.mncore.l1load.v4f32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
