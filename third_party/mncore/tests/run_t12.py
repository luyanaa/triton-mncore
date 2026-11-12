#!/usr/bin/env python3
"""T12: the blocked GEMM -- K/N slicing of one tt.dot into gemm.tile blocks.

`t12_sliced_gemm.mlir` is the TTGIR the frontend produces for a 128x16x16
`tl.dot` (plus an explicit accumulator). M = 128 is one gemm.tile grid, so
KB = K/4 = 4 and NB = N/4 = 4, and the pass must emit:

  * one DMA per buffer plane: 8*KB planes for A, 2*KB*NB for B, 4*NB for C,
    each a 64-longword copy (MV moves 64 longwords at a time, 22.33);
  * KB*NB gemm.tile ops, with the K chain running through y and the face
    alternating 0/1;
  * B arrives through `mncore.x.load` -- the matrix x role, whose two planes
    fill the slot's consecutive longwords 0 and 1 (design doc 22.35) -- and
    never through `mncore.vec.load`, whose `ll` pair no matrix operand reads;
  * one slab store + four dma.store per N block, into the tt.store's buffer
    (which need not be the accumulator's);
  * no ordinary load / convert_layout / store for the dot's operands or
    result -- those are dead once the grid exists;
  * every intrinsic name it emits resolves in LLVM's own intrinsic table:
    `mlir-translate` is run as a local guard, because a `.td` def
    `int_mncore_<a>_<b>` registers as llvm.mncore.<a>.<b> and a name carrying
    an underscore would parse in the C++ pass yet fail there (22.35.2).

`t12_zero_acc_reject.mlir` pins the one operand it cannot synthesise: a
compiler-made zero accumulator is rejected, because y comes from a PE slab and
the pass has no immediate-to-PE-data primitive.

`t12b_k8x16.mlir` / `t12b_k16x8.mlir` pin that KB and NB come from the logical
shape independently (2x4 and 4x2 grids), and `t12b_two_dots.mlir` pins that two
dots with their own buffers each get their own grid and regions.

`run_e2e_dot.py` (opt-in, --remote) runs the 128x16x16 kernel on gpfn3 and
checks the data path: with the emulator's `dst := y`, C must come back
unchanged through DMA -> slab -> gemm.tile -> slab -> DMA. The product is not
checked (G2 open).

Structural only -- G2 (fmfma numerics) is open.

Usage: PYTHONPATH=python python run_t12.py
"""

import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
TRITON_OPT = HERE.parents[2].parent / "triton-mncore-build" / "bin" / "triton-opt"
# Optional: the local LLVM build, used to check that every intrinsic name the
# pass emits is one LLVM actually registered (see the note in 22.35.2).
LLVM = pathlib.Path(os.environ.get(
    "LLVM_BIN", "/Users/yanlu/Documents/llvm-mncore/build-mlir/bin"))


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
            '"mncore.x.load"': kb * nb,
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
        # B is a matrix role: it must not come from the vector form, whose
        # planes sit at longwords 0 and 2 instead of the consecutive pair the
        # word-width x operand reads (design doc 22.35).
        if '"mncore.vec.load"' in mnir:
            failures.append("t12: B was loaded with mncore.vec.load (the `ll` "
                            "pair), not mncore.x.load")

        # The grid must also survive the mncore->LLVM conversion: each
        # gemm.tile is two matwrites plus a matvec (T11's sequence).
        r2 = run("t12_sliced_gemm.mlir", "--tritongpu-to-mncore",
                 "--convert-mncore-to-llvm")
        if r2.returncode:
            failures.append("t12: mncore-to-llvm failed: "
                            + (r2.stdout + r2.stderr)[:400])
        else:
            for needle in ("llvm.mncore.l1load.v8f32", "llvm.mncore.l1store.v8f32",
                           "llvm.mncore.l1xload.v4f32", "llvm.mncore.mv"):
                if needle not in r2.stdout:
                    failures.append(f"t12: LLVM IR lacks {needle}")
            # Every emitted intrinsic name must resolve in LLVM's own table:
            # `def int_mncore_<a>_<b>` registers as llvm.mncore.<a>.<b> (each
            # underscore becomes a dot), so a name containing `_` still parses
            # in the C++ pass but fails here. Skipped if the build is absent.
            translate = LLVM / "mlir-translate"
            if translate.exists():
                r3 = subprocess.run([str(translate), "--mlir-to-llvmir"],
                                    input=r2.stdout, capture_output=True,
                                    text=True)
                if r3.returncode:
                    failures.append(
                        "t12: mlir-translate could not resolve an intrinsic "
                        "name: " + (r3.stdout + r3.stderr)[:400])
        if not failures:
            print(f"PASS  t12: 128x16x16 -> {kb}x{nb} gemm.tile grid "
                  f"({8*kb + 2*kb*nb + 4*nb} plane DMAs, K chained, faces 0/1)")

    # B. KB and NB are derived from the logical shape independently.
    for name, kb, nb in (("t12b_k8x16.mlir", 2, 4),
                         ("t12b_k16x8.mlir", 4, 2)):
        r = run(name, "--tritongpu-to-mncore")
        if r.returncode:
            failures.append(name + ": " + (r.stdout + r.stderr)[:400])
            continue
        mnir = r.stdout
        want = {
            '"mncore.gemm.tile"': kb * nb,
            '"mncore.dma.store"': 4 * nb,
            '"mncore.slab.store"': nb,
            '"mncore.x.load"': kb * nb,
            '"mncore.slab.load"': kb * nb * 2 + nb,
            '"mncore.dma"': 8 * kb + 2 * kb * nb + 4 * nb,
            '<{face = 0 : i64}>': kb * nb // 2,
            '<{face = 1 : i64}>': kb * nb // 2,
        }
        for needle, count in want.items():
            got = mnir.count(needle)
            if got != count:
                failures.append(f"{name}: {needle} x{got}, want x{count}")
        if not any(f.startswith(name) for f in failures):
            print(f"PASS  {name}: KB={kb} NB={nb} grid "
                  f"({kb*nb} gemm.tile, K chained {kb} deep)")

    # C. Two dots, each with its own buffers, get their own grids.
    r = run("t12b_two_dots.mlir", "--tritongpu-to-mncore")
    if r.returncode:
        failures.append("t12b_two_dots.mlir: " + (r.stdout + r.stderr)[:400])
    else:
        mnir = r.stdout
        kb = nb = 2      # each dot is 128x8x8
        want = {
            '"mncore.gemm.tile"': 2 * kb * nb,
            '"mncore.dma.store"': 2 * 4 * nb,
            '"mncore.slab.store"': 2 * nb,
            '"mncore.x.load"': 2 * kb * nb,
            '"mncore.slab.load"': 2 * (kb * nb * 2 + nb),
            '"mncore.dma"': 2 * (8 * kb + 2 * kb * nb + 4 * nb),
        }
        for needle, count in want.items():
            got = mnir.count(needle)
            if got != count:
                failures.append(f"two_dots: {needle} x{got}, want x{count}")
        # Dot 1: A=arg0, B=arg1, acc=arg6, writeback=arg2.
        # Dot 2: A=arg3, B=arg4, acc=arg7, writeback=arg5.
        for arg, planes in ((0, 8 * kb), (1, 2 * kb * nb), (6, 4 * nb),
                            (3, 8 * kb), (4, 2 * kb * nb), (7, 4 * nb)):
            got = mnir.count(f"src_arg = {arg} : i32")
            if got != planes:
                failures.append(f"two_dots: {planes} planes from arg {arg}, "
                                f"got {got}")
        for arg in (2, 5):
            got = mnir.count(f"dr_arg = {arg} : i32")
            if got != 4 * nb:
                failures.append(f"two_dots: {4*nb} writeback planes to arg "
                                f"{arg}, got {got}")
        if not any(f.startswith("two_dots") for f in failures):
            print("PASS  t12b_two_dots.mlir: two independent grids, "
                  "one set of buffers each")

    # D. The zero accumulator must be named, not lowered.
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
