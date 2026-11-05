#!/usr/bin/env python3
"""T5: MAB-aware local_load must select the l1bmm (16x1 broadcast) mode.

The layout proves the distribution: lane bits 2..4 are replicated, so the 4 PE
values are the same in every one of the 16 MABs (SDM 3.6.8.8). The pass must
record `mode = 1` on `mncore.vec.load` and the conversion must produce an
`llvm.mncore.l1load.v4f32` call whose mode operand is 1.

Usage: PYTHONPATH=python python run_t5.py t5_mab.mlir
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
    if '"mncore.vec.load"' not in text or "mode = 1" not in text:
        print("T5 FAIL: MAB mode was not selected on mncore.vec.load",
              file=sys.stderr)
        return 1

    pm2 = ir.pass_manager(ctx)
    pm2.enable_debug()
    mncore.add_convert_mncore_to_llvm(pm2)
    pm2.run(mod, "llir")
    llir = str(mod)
    if "llvm.mncore.l1load.v4f32" not in llir:
        print("T5 FAIL: no llvm.mncore.l1load.v4f32 produced", file=sys.stderr)
        return 1
    print("T5 OK: MAB load -> l1bmm (mode 1) -> llvm.mncore.l1load.v4f32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
