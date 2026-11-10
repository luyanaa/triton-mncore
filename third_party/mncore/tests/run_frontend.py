#!/usr/bin/env python3
"""Higher-level Triton regression for the mncore backend.

  1. Drive the real Triton frontend (ASTSource -> TTIR -> TTGIR) on an
     elementwise add kernel.
  2. Run `tritongpu-to-mncore` on that TTGIR and assert the full add kernel
     lowers to `.vsm`: pointer args -> two-plane DMA (`l2bmb`), a distributed
     `l1bmd` load and an `l2bm@0` point-transfer writeback.
  3. The same for a rank-2 (2-D) kernel and for a two-tile kernel: each tile
     gets its own DMA pair, its own L1BM windows and its own writeback.
  4. Reject what the 256xf32 global path cannot express: a masked tile load
     (no per-element predicate) and a 512-element tile (8 f32 per PE).
  5. Lower the MAB-aware TTGIR fixtures (Broadcast/MAB/MAB4) end-to-end to
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


@triton.jit
def add2dk(a_ptr, b_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr):
    rm = tl.arange(0, R)
    rn = tl.arange(0, C)
    off = rm[:, None] * C + rn[None, :]
    x = tl.load(a_ptr + off)
    y = tl.load(b_ptr + off)
    tl.store(out_ptr + off, x + y)


@triton.jit
def two_tilesk(a_ptr, b_ptr, o_ptr, c_ptr, d_ptr, p_ptr,
               R: tl.constexpr, C: tl.constexpr):
    rm = tl.arange(0, R)
    rn = tl.arange(0, C)
    off = rm[:, None] * C + rn[None, :]
    x = tl.load(a_ptr + off)
    y = tl.load(b_ptr + off)
    tl.store(o_ptr + off, x + y)
    u = tl.load(c_ptr + off)
    v = tl.load(d_ptr + off)
    tl.store(p_ptr + off, u * v)


@triton.jit
def maskedk(a_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
    rm = tl.arange(0, R)
    rn = tl.arange(0, C)
    off = rm[:, None] * C + rn[None, :]
    x = tl.load(a_ptr + off, mask=off < R * C, other=0.0)
    tl.store(o_ptr + off, x)


def make_ttgir(fn, signature, constexprs, num_warps=2) -> str:
    target = GPUTarget("mncore", "mncore2", 1)
    backend = MncoreBackend(target)
    opts = backend.parse_options(
        {"num_warps": num_warps, "num_ctas": 1, "num_stages": 1})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    src = ASTSource(fn, signature=signature, constexprs=constexprs)
    meta = {}
    mod = src.make_ir(target, opts, backend.get_codegen_implementation(opts),
                      backend.get_module_map(), ctx)
    mod = backend.make_ttir(mod, meta, opts)
    mod = backend.make_ttgir(mod, meta, opts)
    return str(mod)


def to_mncore(name, ttgir):
    """TTGIR text -> (mncore text, error)."""
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(ttgir)
        path = f.name
    r = run([str(TRITON_OPT), path, "--tritongpu-to-mncore"])
    if r.returncode != 0:
        return None, (f"{name}: tritongpu-to-mncore failed: "
                      + (r.stdout + r.stderr)[:400])
    return r.stdout, None


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def lower_text(name, text, expects, counts=None):
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
    if counts:
        bad = {k: {"want": v, "got": r4.stdout.count(k)}
               for k, v in counts.items() if r4.stdout.count(k) != v}
        if bad:
            return f"{name}: instruction counts do not match: {bad}"
    return None


def lower_fixture(name, expect):
    p = HERE / name
    r = run([str(TRITON_OPT), str(p), "--tritongpu-to-mncore"])
    if r.returncode != 0:
        return f"{name}: tritongpu-to-mncore failed: {r.stderr[:300]}"
    return lower_text(name, r.stdout, [expect])


def main() -> int:
    failures = []

    ttgir = make_ttgir(addk,
                       {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32",
                        "BLOCK": "constexpr"},
                       {"BLOCK": 256})
    if "tt.load" not in ttgir or "ttg.num-warps" not in ttgir:
        failures.append("frontend did not produce the expected TTGIR")
    else:
        mn, err = to_mncore("add-kernel", ttgir)
        if err:
            failures.append(err)
        else:
            # Pointer args lower to static DRAM bases + two-plane DMA, a
            # distributed l1bmd load, and an l2bm@ point-transfer writeback.
            err = lower_text("add-kernel", mn,
                             ["l2bmb ", "l1bmd ", "l2bm@0 "],
                             {"l2bmb ": 4, "l1bmd ": 6, "l2bm@": 2})
            if err:
                failures.append(err)
            else:
                print("PASS  frontend TTIR/TTGIR + DMA + l1bmd/l2bm@ @" +
                      ".vsm")

    # --- rank-2 (2-D) kernels ---------------------------------------------
    # A 16x16 tile is the same 256 f32 the 1-D kernel moves, so it must
    # produce the same instruction mix: the rank only changes how the row-
    # major offsets are spelled in TTGIR, not what is stored in DRAM.
    two_d_sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32",
                 "R": "constexpr", "C": "constexpr"}
    mn, err = to_mncore("add2d",
                        make_ttgir(add2dk, two_d_sig, {"R": 16, "C": 16}))
    if err:
        failures.append(err)
    else:
        err = lower_text("add2d", mn, ["l2bmb ", "l1bmd ", "l2bm@0 "],
                         {"l2bmb ": 4, "l1bmd ": 6, "l2bm@": 2})
        if err:
            failures.append(err)
        else:
            print("PASS  2-D 16x16 kernel lowers like the 1-D add")

    # Two tiles per kernel: each tile is an independent DMA pair / L1BM
    # window / writeback, and the second tile must not reuse the first
    # tile's addresses. This is the multi-tile form the 256-f32 path admits
    # (a second tile at `base_ptr + N` would not: the offsets of a tile must
    # be exactly 0..N-1 relative to its own buffer argument).
    six_sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "c_ptr": "*fp32", "d_ptr": "*fp32", "p_ptr": "*fp32",
               "R": "constexpr", "C": "constexpr"}
    mn, err = to_mncore("two-tiles",
                        make_ttgir(two_tilesk, six_sig, {"R": 16, "C": 16}))
    if err:
        failures.append(err)
    else:
        err = lower_text("two-tiles", mn, ["l2bmb ", "l1bmd ", "l2bm@0 "],
                         {"l2bmb ": 8, "l1bmd ": 12, "l2bm@": 4})
        if err:
            failures.append(err)
        else:
            print("PASS  two 2-D tiles per kernel: 8 DMA, 12 l1bmd, 4 l2bm@")

    # --- rejections the 256xf32 global path cannot express -----------------
    for name, fn, sig, cexp, want in [
        ("masked-load", maskedk,
         {"a_ptr": "*fp32", "o_ptr": "*fp32",
          "R": "constexpr", "C": "constexpr"},
         {"R": 16, "C": 16}, "masked tile loads are not lowerable"),
        ("512-element tile", add2dk, two_d_sig, {"R": 32, "C": 16},
         "supports a 256xf32 tile"),
    ]:
        mn, err = to_mncore(name, make_ttgir(fn, sig, cexp))
        if err is None:
            failures.append(f"{name}: accepted, but must be rejected")
        elif want not in err:
            failures.append(f"{name}: rejection lacks {want!r}: {err[:300]}")
        else:
            print(f"PASS  {name} rejected: {want}")

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
