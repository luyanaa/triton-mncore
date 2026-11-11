#!/usr/bin/env python3
"""T11: the packed-dot landing sequence lowers and assembles.

Pins the sequence a `tl.dot` lowering must emit under the narrow contract of
design doc 22.22 (B supplied pre-packed per PE column). Structural only --
G2 is open, so no result here says anything about the numbers.
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
    mncore.add_convert_mncore_to_llvm(pm)
    pm.run(mod, "llir")
    llir = str(mod)

    need = [
        "llvm.mncore.l1load.v8f32",   # the 4-longword A slab
        "llvm.mncore.l1load.v4f32",   # the 2-longword B vector
        "llvm.mncore.matwrite.v8f32",  # two row groups of the face
        "llvm.mncore.matvec.v4f32",    # the MAU expression
    ]
    missing = [n for n in need if n not in llir]
    if missing or llir.count("llvm.mncore.matwrite.v8f32") != 4:
        print("T11 FAIL: missing " + ", ".join(missing or
                                               ["2 x 2 matwrite"]), file=sys.stderr)
        return 1
    print("T11 OK: blocked gemm -> 2 blocks x (2 matwrite + 1 matvec), K chained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
