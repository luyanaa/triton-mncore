#!/usr/bin/env python3
"""Higher-level Triton regression for the mncore backend.

  1. Drive the real Triton frontend (ASTSource -> TTIR -> TTGIR) on an
     elementwise add kernel.
  2. Run `tritongpu-to-mncore` on that TTGIR and assert the full add kernel
     lowers to `.vsm`: pointer args -> two-plane DMA (`l2bmb`), a distributed
     `l1bmd` load and an `l2bm@0` point-transfer writeback.
  3. Lower the MAB-aware TTGIR fixtures (Broadcast/MAB/MAB4) end-to-end to
     .vsm and assert the physical L1BM instruction (l1bmp/l1bmm/l1bmm4).

Usage: PYTHONPATH=python python run_frontend.py
"""

import pathlib
import subprocess
import sys
import tempfile

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.mncore.compiler import MncoreBackend
from triton.compiler.compiler import ASTSource

HERE = pathlib.Path(__file__).resolve().parent
BUILD = pathlib.Path(__file__).resolve().parents[3].parent / "triton-mncore-build"
LLVM = pathlib.Path("/Users/yanlu/Documents/llvm-mncore/build-mlir/bin")
TRITON_OPT = BUILD / "bin" / "triton-opt"


@triton.jit
def addk(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(out_ptr + offs, x + y)


def make_ttgir() -> str:
    target = GPUTarget("mncore", "mncore2", 1)
    backend = MncoreBackend(target)
    opts = backend.parse_options(
        {"num_warps": 2, "num_ctas": 1, "num_stages": 1})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    src = ASTSource(
        addk,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32",
                   "BLOCK": "constexpr"},
        constexprs={"BLOCK": 256})
    meta = {}
    mod = src.make_ir(target, opts, backend.get_codegen_implementation(opts),
                      backend.get_module_map(), ctx)
    mod = backend.make_ttir(mod, meta, opts)
    mod = backend.make_ttgir(mod, meta, opts)
    return str(mod)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def lower_text(name, text, expects):
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(text)
        mnir = f.name
    r2 = run([str(LLVM / "mlir-opt"), mnir, "--convert-mncore-to-llvm"])
    if r2.returncode != 0:
        return f"{name}: convert-mncore-to-llvm failed: {r2.stderr[:300]}"
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(r2.stdout)
        ll = f.name
    r3 = run([str(LLVM / "mlir-translate"), ll, "--mlir-to-llvmir"])
    if r3.returncode != 0:
        return f"{name}: mlir-translate failed: {r3.stderr[:300]}"
    with tempfile.NamedTemporaryFile("w", suffix=".ll", delete=False) as f:
        f.write(r3.stdout)
        llfile = f.name
    r4 = run([str(LLVM / "llc"), "-mtriple=mncore2", llfile, "-o", "-"])
    if r4.returncode != 0:
        return f"{name}: llc failed: {r4.stderr[:300]}"
    missing = [e for e in expects if e not in r4.stdout]
    if missing:
        return (f"{name}: expected {missing} in .vsm, got: "
                f"{r4.stdout[:300]}")
    return None


def lower_fixture(name, expect):
    p = HERE / name
    r = run([str(TRITON_OPT), str(p), "--tritongpu-to-mncore"])
    if r.returncode != 0:
        return f"{name}: tritongpu-to-mncore failed: {r.stderr[:300]}"
    return lower_text(name, r.stdout, [expect])


def main() -> int:
    failures = []

    ttgir = make_ttgir()
    if "tt.load" not in ttgir or "ttg.num-warps" not in ttgir:
        failures.append("frontend did not produce the expected TTGIR")
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
            f.write(ttgir)
            path = f.name
        r = run([str(TRITON_OPT), path, "--tritongpu-to-mncore"])
        if r.returncode != 0:
            failures.append("tritongpu-to-mncore rejected the add kernel: "
                            + r.stderr[:300])
        else:
            # Pointer args lower to static DRAM bases + two-plane DMA, a
            # distributed l1bmd load, and an l2bm@ point-transfer writeback.
            err = lower_text("add-kernel", r.stdout,
                             ["l2bmb ", "l1bmd ", "l2bm@0 "])
            if err:
                failures.append(err)
            else:
                print("PASS  frontend TTIR/TTGIR + DMA + l1bmd/l2bm@ @" +
                      ".vsm")

    for name, expect in [
        ("t7_broadcast.mlir", "l1bmp"),
        ("t5_mab.mlir", "l1bmm "),
        ("t6_mab4.mlir", "l1bmm4 "),
        ("t3_distributed.mlir", "l1bmd $lb0"),
        ("t4_roundtrip.mlir", "l1bmd $ls4v $lb64"),
    ]:
        err = lower_fixture(name, expect)
        if err:
            failures.append(err)
        else:
            print(f"PASS  {name} -> {expect.strip()}")

    if failures:
        for f in failures:
            print("FAIL  " + f, file=sys.stderr)
        return 1
    print("FRONTEND REGRESSION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
