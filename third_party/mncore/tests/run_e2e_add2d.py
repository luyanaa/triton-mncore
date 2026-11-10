#!/usr/bin/env python3
"""M2: two-tile rank-2 (2-D) Triton kernel end-to-end with a CPU golden.

  @triton.jit add2d(A0,B0,O0, A1,B1,O1, R=16, C=16)
      -> TTIR/TTGIR (2-D `rm[:, None] * C + rn[None, :]` tiles)
      -> mncore (2 x DMA + distributed l1bmd + vec op + l2bm@ writeback)
      -> LLVM IR -> .vsm -> assemble3 (hazard checker) -> gpfn3 -> DRAM dump
      -> compare every output longword with the CPU golden.

Why two tiles: a single tile is already covered by run_e2e_add.py, and the
rank-2 form lowers to the same instruction stream, so the interesting new
property is that several tiles coexist in one kernel -- separate buffers,
separate DMA tags, separate L1BM windows, separate writebacks. If any of
those addresses were reused, tile 1 would silently overwrite tile 0 in DRAM,
which the element-for-element golden catches.

The physical DRAM order is the compiler's canonical plane-major order: each
plane is 64 longwords holding two f32 per longword, so element 2j and 2j+1 of
the logical row-major tile land in the low and high half of longword j. The
CPU side builds the expected longwords the same way, so the comparison is
element for element in physical order (a permutation fails).

Environment: MNRT_HOST (default suzhou), ASSEMBLE3, GPFN3, TRITON_BUILD_DIR,
LLVM_BIN. Requires ssh access to the emulator host.
"""

import os
import pathlib
import re
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

import mncore_fixup

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

R, C = 16, 16                       # 16x16 f32 tile
N = R * C                           # 256 f32 elements per tile
PLANE_LONGWORDS = N // 4            # 64 longwords per plane, 2 planes

# Six buffers of 2 planes each, laid out back to back in longwords.
BASE = {"a0": 0, "b0": 128, "o0": 256, "a1": 384, "b1": 512, "o1": 640}
ARGS = ["a0", "b0", "o0", "a1", "b1", "o1"]      # kernel argument order
OUTPUTS = ["o0", "o1"]


@triton.jit
def add2dk(a_ptr, b_ptr, o_ptr, c_ptr, d_ptr, p_ptr,
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
        add2dk,
        signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
                   "c_ptr": "*fp32", "d_ptr": "*fp32", "p_ptr": "*fp32",
                   "R": "constexpr", "C": "constexpr"},
        constexprs={"R": R, "C": C})
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
    in_sidecar = False
    for line in r4.stdout.splitlines():
        s = line.strip()
        if not s or s.startswith("."):
            continue
        if s.startswith("#"):
            # Keep the per-line fixup comments, the kernel entry markers (the
            # fixup primary key is (kernel entry, site)), and the whole
            # sidecar block (whose entry lines do not themselves mention
            # "@mncore.fixup").
            if "@mncore.fixup.begin" in s:
                in_sidecar = True
            if in_sidecar or "@mncore.fixup" in s or "@mncore.kernel" in s:
                lines.append(s)
            if "@mncore.fixup.end" in s:
                in_sidecar = False
            continue
        if s.endswith(":") or s.split()[0].endswith(":"):
            continue  # drop labels/directives
        lines.append(s)
    return "\n".join(lines) + "\n"


def dset_dram(addr: int, value: int) -> str:
    spec = 9 | (addr << 4) | (1 << 21)
    return f"d set $d{addr} 1 {value:016x}"


def dump_groups(dump: str, addr: int, count: int):
    """Collect the (n0..n3) DRAM groups read back by one `d get` command."""
    groups = {}
    pat = re.compile(
        r"DEBUG-DRAM\((n\d+),(\d+)\):.*?v:(0x[0-9a-fA-F]+)\) "
        rf"#d get \$d{addr} {count}\b")
    for m in pat.finditer(dump):
        groups.setdefault(m.group(1), []).append(int(m.group(3), 16))
    return groups


def program_and_golden():
    """The kernel's preload commands and the CPU golden for both tiles.

    Shared with run_e2e_mnrt.py so the two execution paths (direct
    assemble3+gpfn3, and through the project runtime) compare against exactly
    the same data.
    """
    golden = {}
    preload = []
    for tile, op in (("o0", "add"), ("o1", "mul")):
        tag = tile[1]                       # "0" / "1"
        vals = []
        for j in range(2 * PLANE_LONGWORDS):
            a = pack2(1.0 + j, 2.0 + j)
            b = pack2(100.0 + j, 200.0 + j)
            la = (1.0 + j, 2.0 + j)
            lb = (100.0 + j, 200.0 + j)
            c = (f32(la[0] + lb[0]) if op == "add" else f32(la[0] * lb[0]),
                 f32(la[1] + lb[1]) if op == "add" else f32(la[1] * lb[1]))
            vals.append(pack2(*c))
            preload.append(dset_dram(BASE["a" + tag] + j, a))
            preload.append(dset_dram(BASE["b" + tag] + j, b))
        golden[tile] = vals
    return preload, golden


def relocated_kernel():
    """Compile the kernel, guard the writeback, bind every buffer argument."""
    vsm = vsm_from_ttgir(ttgir())
    # Instruction-level guard for the writeback path. `l2bmd` (collect) builds
    # L2BM from 8-longword chunks taken from all eight L1Bs; a Triton
    # distributed tile is replicated, so the collect interleaves eight
    # identical copies and destroys the plane. The store must use the
    # single-L1B point transfer `l2bm@<l1b>` instead -- for both tiles.
    if vsm.count("l2bm@0 ") != 4 or "l2bmd " in vsm:
        raise RuntimeError("writeback is not two point-transfer pairs\n" + vsm)
    # Every buffer argument is relocatable: each DRAM operand is a placeholder
    # 0 plus a fixup record. This host acts as the loader and binds arg i to
    # the DRAM base its golden data was preloaded at (kernel ABI §13.3).
    bindings = {i: BASE[name] * 8 for i, name in enumerate(ARGS)}
    vsm, applied = mncore_fixup.relocate(vsm, bindings)
    # 2 tiles x (2 loads + 1 store) x 2 longword planes.
    if len(applied) != 12:
        raise RuntimeError(
            f"expected 12 relocated operands (2 tiles x 3 transfers x 2 "
            f"planes), got {len(applied)}")
    return vsm


def full_body():
    """preload | kernel | d get, the single program both backends execute."""
    preload, golden = program_and_golden()
    body = "\n".join(preload) + "\n" + relocated_kernel()
    for tile in OUTPUTS:
        body += f"d get $d{BASE[tile]} {2 * PLANE_LONGWORDS}\n"
    return body, golden


def main() -> int:
    try:
        body, golden = full_body()
    except mncore_fixup.RelocError as e:
        print(f"E2E FAIL: relocation: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"E2E FAIL: {e}", file=sys.stderr)
        return 1

    with tempfile.NamedTemporaryFile("w", suffix=".vsm", delete=False) as f:
        f.write(body)
        local = f.name

    stage = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", HOST, "cat > /tmp/mncore_e2e_add2d.vsm"],
        stdin=open(local, "rb"))
    if stage.returncode != 0:
        print("E2E FAIL: staging", file=sys.stderr)
        return 1

    r = run(["ssh", "-o", "BatchMode=yes", HOST,
             f"{SDE} -hsw -- {ASM} --output-file /tmp/mncore_e2e_add2d.asm "
             f"/tmp/mncore_e2e_add2d.vsm"])
    if r.returncode != 0 or "racy" in r.stderr or "Error" in r.stderr:
        print("E2E FAIL: assemble3\n" + (r.stdout + r.stderr)[:800], file=sys.stderr)
        return 1

    r = run(["ssh", "-o", "BatchMode=yes", HOST,
             f"{SDE} -hsw -- {EMU} -i /tmp/mncore_e2e_add2d.asm "
             f"-d /tmp/mncore_e2e_add2d.dmp"])
    if r.returncode != 0:
        print("E2E FAIL: gpfn3\n" + (r.stdout + r.stderr)[:800], file=sys.stderr)
        return 1

    dump = run(["ssh", "-o", "BatchMode=yes", HOST,
                "cat /tmp/mncore_e2e_add2d.dmp"]).stdout
    for tile in OUTPUTS:
        groups = dump_groups(dump, BASE[tile], 2 * PLANE_LONGWORDS)
        want = golden[tile]
        if len(groups) != 4 or any(len(v) != len(want) for v in groups.values()):
            print(f"E2E FAIL: {tile}: expected 4 groups of {len(want)} values, "
                  f"got { {k: len(v) for k, v in groups.items()} }",
                  file=sys.stderr)
            return 1
        for g, vals in sorted(groups.items()):
            bad = next((i for i, (x, y) in enumerate(zip(vals, want))
                        if x != y), None)
            if bad is not None:
                print(f"E2E FAIL: {tile}/{g} differs at index {bad}: "
                      f"{vals[bad]:#x} vs {want[bad]:#x}", file=sys.stderr)
                return 1
    print(f"E2E ADD2D OK: 2 tiles x 4 x {2 * PLANE_LONGWORDS} DRAM longwords "
          f"(add, mul) match the CPU golden element for element")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
