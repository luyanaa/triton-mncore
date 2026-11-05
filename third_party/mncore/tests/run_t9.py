#!/usr/bin/env python3
"""T9: the tt.dot / GEMM contract on MN-Core.

Two halves:

  A. Frontend. A real `@triton.jit` 32x32x32 `tl.dot` kernel must reach TTGIR
     with the mncore backend. That pins the two `MncoreOptions` /
     `get_codegen_implementation` hooks the frontend needs (`sanitize_overflow`
     and the dot precision policy fields, `min_dot_size`). The emitted layout
     is the contract's starting point: `#blocked` with sizePerThread [4, 4]
     gives every thread a 4x4 output tile, and `#ttg.dot_op<parent=#blocked>`
     replicates the whole K dimension into registers.

  B. Gate. `tritongpu-to-mncore` must reject every dot with the clause of the
     contract that actually fails, and never lower one. The MAU has no
     matrix-matrix instruction (SDM 3.6.9.1: mfma and mmul only, one matrix
     operand each); a kernel GEMM has to be mwrite + one fmfma per 4 columns
     of B. Lowering a `tl.dot` through the blocked FMA dot-operand layout
     would silently compute the wrong operand placement, so it must fail.

Usage: PYTHONPATH=python python run_t9.py
"""

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
TRITON_OPT = HERE.parents[2].parent / "triton-mncore-build" / "bin" / "triton-opt"

# Rejection contract: (fixture, required needles, description).
CASES = [
    ("t9_dot_contract.mlir",
     ["is not lowered yet",
      "one fmfma per 4 columns of B",
      "K one element per PE",
      "the MAU GEMM sequence is pending"],
     "the pending GEMM sequence and the operand-placement mismatch"),
    ("t9_dot_dtype_reject.mlir",
     ["is f32 only",
      "hmfma",
      "u|d row half-select"],
     "the MAU precision forms (h / d)"),
    ("t9_dot_shape_reject.mlir",
     ["does not tile the MAU",
      "multiples of 8, 4 and 4"],
     "the 8-row face / 4-column N / K=4 tiling"),
]


def frontend_ttgir() -> str:
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.mncore.compiler import MncoreBackend
    from triton.compiler.compiler import ASTSource

    @triton.jit
    def dotk(a_ptr, b_ptr, c_ptr):
        rm = tl.arange(0, 32)
        rn = tl.arange(0, 32)
        rk = tl.arange(0, 32)
        a = tl.load(a_ptr + rm[:, None] * 32 + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * 32 + rn[None, :])
        c = tl.dot(a, b)
        tl.store(c_ptr + rm[:, None] * 32 + rn[None, :], c)

    target = GPUTarget("mncore", "mncore2", 1)
    backend = MncoreBackend(target)
    opts = backend.parse_options(
        {"num_warps": 2, "num_ctas": 1, "num_stages": 1})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    src = ASTSource(
        dotk,
        signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32"})
    meta = {}
    mod = src.make_ir(target, opts, backend.get_codegen_implementation(opts),
                      backend.get_module_map(), ctx)
    mod = backend.make_ttir(mod, meta, opts)
    mod = backend.make_ttgir(mod, meta, opts)
    return str(mod)


def main() -> int:
    failures = []

    # A. The frontend must accept tl.dot and hand over a dot with the FMA
    #    dot-operand encoding (no GPU mma layout exists for this target).
    try:
        ttgir = frontend_ttgir()
    except Exception as exc:  # noqa: BLE001 - report any frontend failure
        ttgir = ""
        failures.append(f"frontend: tl.dot did not reach TTGIR: {exc}")
    if ttgir:
        needs = ["tt.dot", "#ttg.dot_op<{opIdx = 0", "#ttg.dot_op<{opIdx = 1",
                 "sizePerThread = [4, 4]"]
        missing = [n for n in needs if n not in ttgir]
        if missing:
            failures.append(f"frontend TTGIR lacks {missing}")
        else:
            print("PASS  tl.dot 32x32x32 -> TTGIR (dot_op A/B, 4x4 C tile)")

    # B. Every dot must be rejected with its contract clause.
    for name, needles, what in CASES:
        r = subprocess.run(
            [str(TRITON_OPT), str(HERE / name), "--tritongpu-to-mncore"],
            capture_output=True, text=True)
        out = r.stdout + r.stderr
        if r.returncode == 0:
            failures.append(f"{name}: accepted, but a dot must not lower")
            continue
        missing = [n for n in needles if n not in out]
        if missing:
            failures.append(f"{name}: rejection lacks {missing}: {out[:400]}")
            continue
        print(f"PASS  {name} rejected: {what}")

    if failures:
        for f in failures:
            print("T9 FAIL: " + f, file=sys.stderr)
        return 1
    print("T9 OK: tt.dot reaches TTGIR and is gated by the MAU contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
