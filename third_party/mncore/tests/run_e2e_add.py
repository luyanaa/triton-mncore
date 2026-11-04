#!/usr/bin/env python3
"""M1: Triton Vector Add end-to-end with a CPU golden.

  @triton.jit add(A, B, C) -> TTIR/TTGIR -> mncore (DMA + vec ops) ->
  LLVM IR -> .vsm -> assemble3 (hazard checker) -> gpfn3 -> DRAM dump
  -> compare every output longword with the CPU golden.

The host uses the compiler's canonical plane-major physical order for A/B/C;
because the kernel is elementwise, any consistent permutation is valid and
the golden is computed in the same order.

Environment: MNRT_HOST (default suzhou), ASSEMBLE3, GPFN3, TRITON_BUILD_DIR,
LLVM_BIN. Requires ssh access to the emulator host.
"""

import os
import pathlib
import struct
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
BUILD = pathlib.Path(os.environ.get(
    "TRITON_BUILD_DIR", HERE.parents[3] / "triton-mncore-build"))
LLVM = pathlib.Path(os.environ.get(
    "LLVM_BIN", "/Users/yanlu/Documents/llvm-mncore/build-mlir/bin"))
HOST = os.environ.get("MNRT_HOST", "suzhou")
SDE = os.environ.get("SDE",
                     "/opt/sde-external-10.13.1-2026-07-28-lin/sde64")
ASM = os.environ.get("ASSEMBLE3",
                     "/opt/pfn/pfcomp/codegen/build/tools/assemble3")
EMU = os.environ.get("GPFN3",
                     "/opt/pfn/pfcomp/codegen/build/tools/gpfn3_package_main")

N = 256                  # f32 elements per tile
PLANE_LONGWORDS = N // 4  # 64 longwords per plane
BASE_A, BASE_B, BASE_C = 0, 128, 256


@triton.jit
def addk(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(out_ptr + offs, x + y)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def pack2(lo, hi):
    return struct.unpack("<Q", struct.pack("<ff", f32(lo), f32(hi)))[0]


def ttgir() -> str:
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
        constexprs={"BLOCK": N})
    meta = {}
    mod = src.make_ir(target, opts, backend.get_codegen_implementation(opts),
                      backend.get_module_map(), ctx)
    mod = backend.make_ttir(mod, meta, opts)
    mod = backend.make_ttgir(mod, meta, opts)
    return str(mod)


def run(cmd, stdin=None):
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True)


def vsm_from_ttgir(text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(text)
        p = f.name
    r = run([str(BUILD / "bin/triton-opt"), p, "--tritongpu-to-mncore"])
    if r.returncode != 0:
        raise RuntimeError("tritongpu-to-mncore: " + r.stderr[:500])
    r2 = run([str(LLVM / "mlir-opt"), "--convert-mncore-to-llvm"], r.stdout)
    if r2.returncode != 0:
        raise RuntimeError("convert-mncore-to-llvm: " + r2.stderr[:500])
    r3 = run([str(LLVM / "mlir-translate"), "--mlir-to-llvmir"], r2.stdout)
    if r3.returncode != 0:
        raise RuntimeError("mlir-translate: " + r3.stderr[:500])
    with tempfile.NamedTemporaryFile("w", suffix=".ll", delete=False) as f:
        f.write(r3.stdout)
        p = f.name
    r4 = run([str(LLVM / "llc"), "-mtriple=mncore2", p, "-o", "-"])
    if r4.returncode != 0:
        raise RuntimeError("llc: " + r4.stderr[:500])
    lines = []
    for line in r4.stdout.splitlines():
        s = line.strip()
        if not s or s.startswith(".") or s.startswith("#"):
            continue
        if s.endswith(":") or s.split()[0].endswith(":"):
            continue  # drop labels/directives
        lines.append(s)
    return "\n".join(lines) + "\n"


def dset_dram(addr: int, value: int) -> str:
    spec = 9 | (addr << 4) | (1 << 21)
    return f"d set $d{addr} 1 {value:016x}"


def main() -> int:
    golden = []
    preload = []
    for j in range(2 * PLANE_LONGWORDS):
        a = pack2(1.0 + j, 2.0 + j)
        b = pack2(100.0 + j, 200.0 + j)
        c = pack2(1.0 + j + 100.0 + j, 2.0 + j + 200.0 + j)
        golden.append(c)
        preload.append(dset_dram(BASE_A + j, a))
        preload.append(dset_dram(BASE_B + j, b))

    vsm = vsm_from_ttgir(ttgir())
    body = "\n".join(preload) + "\n" + vsm + \
        f"d get $d{BASE_C} {2 * PLANE_LONGWORDS}\n"

    with tempfile.NamedTemporaryFile("w", suffix=".vsm", delete=False) as f:
        f.write(body)
        local = f.name

    stage = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", HOST, "cat > /tmp/mncore_e2e_add.vsm"],
        stdin=open(local, "rb"))
    if stage.returncode != 0:
        print("E2E FAIL: staging", file=sys.stderr)
        return 1

    r = run(["ssh", "-o", "BatchMode=yes", HOST,
             f"{SDE} -hsw -- {ASM} --output-file /tmp/mncore_e2e_add.asm "
             f"/tmp/mncore_e2e_add.vsm"])
    if r.returncode != 0 or "racy" in r.stderr or "Error" in r.stderr:
        print("E2E FAIL: assemble3\n" + (r.stdout + r.stderr)[:800], file=sys.stderr)
        return 1

    r = run(["ssh", "-o", "BatchMode=yes", HOST,
             f"{SDE} -hsw -- {EMU} -i /tmp/mncore_e2e_add.asm "
             f"-d /tmp/mncore_e2e_add.dmp"])
    if r.returncode != 0:
        print("E2E FAIL: gpfn3\n" + (r.stdout + r.stderr)[:800], file=sys.stderr)
        return 1

    dump = run(["ssh", "-o", "BatchMode=yes", HOST,
                "cat /tmp/mncore_e2e_add.dmp"]).stdout
    import re
    # The dump exposes the physical DRAM order, which the emulator interleaves
    # across its four groups (n0..n3). Compare each group as a multiset against
    # the CPU golden with *distinct* inputs: a duplicated, lost, or
    # mis-paired element changes the multiset.
    groups = {}
    for m in re.finditer(
            r"DEBUG-DRAM\((n\d+),(\d+)\):.*?v:(0x[0-9a-fA-F]+)\) #d get \$d256 128",
            dump):
        groups.setdefault(m.group(1), []).append(int(m.group(3), 16))
    if len(groups) != 4 or any(len(v) != len(golden) for v in groups.values()):
        print(f"E2E FAIL: expected 4 groups of {len(golden)} values, got "
              f"{ {k: len(v) for k, v in groups.items()} }", file=sys.stderr)
        return 1
    want = sorted(golden)
    for g, vals in sorted(groups.items()):
        if sorted(vals) != want:
            bad = next((i for i, (a, b) in enumerate(zip(sorted(vals), want))
                        if a != b), None)
            print(f"E2E FAIL: group {g} multiset differs at sorted index {bad}: "
                  f"{sorted(vals)[bad]:#x} vs {want[bad]:#x}", file=sys.stderr)
            return 1
    # Report the physical permutation for the first group (calibration data).
    index_of = {v: i for i, v in enumerate(golden)}
    perm = [index_of[v] for v in groups[sorted(groups)[0]]]
    print(f"E2E ADD OK: 4x{len(golden)} DRAM longwords match the CPU golden; "
          f"physical order is a permutation (first 16: {perm[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
