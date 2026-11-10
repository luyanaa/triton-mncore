#!/usr/bin/env python3
"""T10: rank-2 tiles. The distribution criterion is rank-agnostic.

A 16x16 f32 tile with 4 f32 per PE is Distributed for exactly the same reason
a 256-element 1-D tile is: none of the lane bits are replicated and the element
count is 64 * lanes. The pass must lower it to the *same* two-plane l1bmd
sequence, because the logical row-major order of a 2-D tile is the physical
L1BM/PE order -- that is the contract the 2-D global path relies on.

Half A: `t10_2d.mlir` lowers to `mncore.vec.load` mode 3 read from L1BM 0 and
64, i.e. the two 64-longword planes; the LLVM conversion emits
`llvm.mncore.l1load.v4f32`.

Half B: `t10_2d_narrow_reject.mlir` has 2 f32 per thread (half a PE slot) and
must be rejected rather than silently rounded up to 4.

Usage: PYTHONPATH=python python run_t10.py
"""

import pathlib
import subprocess
import sys

from triton._C.libtriton import ir
from triton._C.libtriton import mncore

HERE = pathlib.Path(__file__).resolve().parent
TRITON_OPT = HERE.parents[2].parent / "triton-mncore-build" / "bin" / "triton-opt"


def lower(path: str):
    """Run mnir, then llir on the same module; return both texts."""
    ctx = ir.context()
    ir.load_dialects(ctx)
    mncore.load_dialects(ctx)
    mod = ir.parse_mlir_module(path, ctx)
    pm = ir.pass_manager(ctx)
    pm.enable_debug()
    mncore.add_tritongpu_to_mncore(pm)
    pm.run(mod, "mnir")
    mnir = str(mod)
    pm2 = ir.pass_manager(ctx)
    pm2.enable_debug()
    mncore.add_convert_mncore_to_llvm(pm2)
    pm2.run(mod, "llir")
    return mnir, str(mod)


def main() -> int:
    failures = []

    mnir, llir = lower(str(HERE / "t10_2d.mlir"))
    print(mnir)
    needs = ['"mncore.vec.load"', "mode = 3", "addr = 0 : i64", "addr = 64 : i64"]
    missing = [n for n in needs if n not in mnir]
    if missing:
        failures.append(f"t10_2d.mlir: lowered form lacks {missing}")
    elif "llvm.mncore.l1load.v4f32" not in llir:
        failures.append("t10_2d.mlir: LLVM conversion did not emit "
                        "llvm.mncore.l1load.v4f32")
    else:
        print("PASS  16x16 tile -> two-plane l1bmd (mode 3, L1BM 0 and 64)")

    # The rank only changes the type, not the instruction stream: lowering the
    # 2-D fixture and the equivalent 1-D one must produce the same bytes.
    flat = (HERE / "t3_distributed.mlir")
    if flat.exists():
        one_d, _ = lower(str(flat))
        strip = lambda s: [ln for ln in s.splitlines()
                           if "mncore." in ln]
        if strip(one_d) != strip(mnir):
            failures.append("2-D lowering differs from the equivalent 1-D one")
        else:
            print("PASS  2-D and 1-D stream of the same tile are identical")

    r = subprocess.run([str(TRITON_OPT), str(HERE / "t10_2d_narrow_reject.mlir"),
                        "--tritongpu-to-mncore"],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode == 0:
        failures.append("t10_2d_narrow_reject.mlir: accepted a half-slot tile")
    elif "neither a full 64-PE distribution" not in out:
        failures.append("t10_2d_narrow_reject.mlir: rejection lacks the reason: "
                        + out[:400])
    else:
        print("PASS  2 f32/thread tile rejected (half a PE slot)")

    if failures:
        for f in failures:
            print("T10 FAIL: " + f, file=sys.stderr)
        return 1
    print("T10 OK: rank-2 tiles lower, half-slot tiles are rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
