#!/usr/bin/env python3
"""T12: the blocked GEMM -- K/N slicing of one tt.dot into gemm.tile blocks.

`t12_sliced_gemm.mlir` is the TTGIR the frontend produces for a 128x16x16
`tl.dot` (plus an explicit accumulator). M = 128 is one gemm.tile grid, so
KB = K/4 = 4 and NB = N/4 = 4, and the pass must emit:

  * one DMA per buffer plane: 8*KB planes for A, 2*KB*NB for B, 4*NB for C,
    each a 64-longword copy (MV moves 64 longwords at a time, 22.33);
  * KB*NB gemm.tile ops, with the K chain running through y and the face
    alternating 0/1;
  * one slab store + four dma.store per N block, into the tt.store's buffer
    (which need not be the accumulator's);
  * no ordinary load / convert_layout / store for the dot's operands or
    result -- those are dead once the grid exists.

`t12_zero_acc_reject.mlir` pins the one operand it cannot synthesise: a
compiler-made zero accumulator is rejected, because y comes from a PE slab and
the pass has no immediate-to-PE-data primitive.

Structural only -- G2 (fmfma numerics) is open.

Usage: PYTHONPATH=python python run_t12.py
"""

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
TRITON_OPT = HERE.parents[2].parent / "triton-mncore-build" / "bin" / "triton-opt"


def run(name: str, *flags: str):
    return subprocess.run([str(TRITON_OPT), str(HERE / name), *flags],
                          capture_output=True, text=True)


def main() -> int:
    failures = []

    # A. The grid.
    r = run("t12_sliced_gemm.mlir", "--tritongpu-to-mncore")
    if r.returncode:
        failures.append("t12_sliced_gemm.mlir: " + (r.stdout + r.stderr)[:600])
    else:
        mnir = r.stdout
        kb = nb = 4
        want = {
            '"mncore.gemm.tile"': kb * nb,
            '"mncore.dma.store"': 4 * nb,
            '"mncore.slab.store"': nb,
            '"mncore.vec.load"': kb * nb,
            '"mncore.slab.load"': kb * nb * 2 + nb,
            '"mncore.dma"': 8 * kb + 2 * kb * nb + 4 * nb,
            '<{face = 0 : i64}>': kb * nb // 2,
            '<{face = 1 : i64}>': kb * nb // 2,
        }
        for needle, count in want.items():
            got = mnir.count(needle)
            if got != count:
                failures.append(f"t12: {needle} x{got}, want x{count}")
        # Buffer attribution: A planes come from arg 0, B from arg 1, the
        # accumulator is read from arg 3 (%acc_ptr) and the writeback goes to
        # the tt.store's buffer, arg 2 (%c_ptr) -- they are distinct here.
        for arg, planes in ((0, 8 * kb), (1, 2 * kb * nb), (3, 4 * nb)):
            got = mnir.count(f"src_arg = {arg} : i32")
            if got != planes:
                failures.append(f"t12: {planes} DMA planes from arg {arg}, "
                                f"got {got}")
        got = mnir.count("dr_arg = 2 : i32")
        if got != 4 * nb:
            failures.append(f"t12: {4 * nb} writeback planes to arg 2, got {got}")
        if "tt.load" in mnir or "tt.store" in mnir or "tt.dot" in mnir:
            failures.append("t12: a Triton op survived the lowering")
        if "ttg.convert_layout" in mnir:
            failures.append("t12: a dead operand relayout survived")

        # The grid must also survive the mncore->LLVM conversion: each
        # gemm.tile is two matwrites plus a matvec (T11's sequence).
        r2 = run("t12_sliced_gemm.mlir", "--tritongpu-to-mncore",
                 "--convert-mncore-to-llvm")
        if r2.returncode:
            failures.append("t12: mncore-to-llvm failed: "
                            + (r2.stdout + r2.stderr)[:400])
        else:
            for needle in ("llvm.mncore.l1load.v8f32", "llvm.mncore.l1store.v8f32",
                           "llvm.mncore.mv"):
                if needle not in r2.stdout:
                    failures.append(f"t12: LLVM IR lacks {needle}")
        if not failures:
            print(f"PASS  t12: 128x16x16 -> {kb}x{nb} gemm.tile grid "
                  f"({8*kb + 2*kb*nb + 4*nb} plane DMAs, K chained, faces 0/1)")

    # B. The zero accumulator must be named, not lowered.
    r = run("t12_zero_acc_reject.mlir", "--tritongpu-to-mncore")
    out = r.stdout + r.stderr
    if r.returncode == 0:
        failures.append("t12_zero_acc_reject.mlir: accepted a zero accumulator")
    else:
        missing = [n for n in ("compiler-synthesised zero accumulator",
                               "mmul", "immediate-to-PE-data")
                   if n not in out]
        if missing:
            failures.append(f"t12_zero_acc_reject.mlir: rejection lacks "
                            f"{missing}: {out[:400]}")
        else:
            print("PASS  t12_zero_acc_reject.mlir rejected: no zero slab source")

    if failures:
        for f in failures:
            print("T12 FAIL: " + f, file=sys.stderr)
        return 1
    print("T12 OK: the dot is sliced into per-block gemm.tile work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
