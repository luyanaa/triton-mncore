#!/usr/bin/env python3
"""T12 end-to-end: run the blocked-GEMM kernel on gpfn3 and check the C path.

The kernel is the T12 fixture, 128x16x16 -> a 4x4 grid of `gemm.tile`. G2 is
open (gpfn3's `fmfma` does not multiply-add; design doc 22.9), so this test
does not check a product. It checks the part that G2 cannot hide: the kernel's
*data path*.

The MAU's observed behaviour is `dst := y`, and the y operand of each block's
K chain is the C block read from the accumulator buffer. So if every address,
plane order, block index and writeback is right, the output buffer must equal
the input C buffer longword for longword. A wrong block stride, plane order,
relocation or writeback shows up immediately as a mismatch; the kernel also
has to assemble (assemble3's hazard checker) and run without trapping.

Scope of the claim: this validates the plumbing and the C read/write
consistency. It says nothing about A/B placement feeding fmfma (no product is
computed), and nothing about numerics -- both stay open.

Usage: PYTHONPATH=python python run_e2e_dot.py [fixture]
"""

import os
import pathlib
import re
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
BUILD = pathlib.Path(os.environ.get(
    "TRITON_BUILD_DIR", HERE.parents[3] / "triton-mncore-build"))
LLVM = pathlib.Path(os.environ.get(
    "LLVM_BIN", "/Users/yanlu/Documents/llvm-mncore/build-mlir/bin"))
HOST = os.environ.get("MNRT_HOST", "suzhou")
SDE = os.environ.get("SDE", "/opt/sde-external-10.13.1-2026-07-28-lin/sde64")
ASM = os.environ.get("ASSEMBLE3",
                     "/opt/pfn/pfcomp/codegen/build/tools/assemble3")
EMU = os.environ.get("GPFN3",
                     "/opt/pfn/pfcomp/codegen/build/tools/gpfn3_package_main")

sys.path.insert(0, str(HERE))
import mncore_fixup  # noqa: E402

# DRAM (longwords). A' and B' are read but never checked, so they are not
# preloaded -- only their addresses have to be in range.
KB = NB = 4
A_WORDS = KB * 512          # 8 planes per K block x 64 longwords
B_WORDS = KB * NB * 128     # 2 planes per block x 64 longwords
C_WORDS = NB * 256          # 4 planes per N block x 64 longwords
BASE_A, BASE_B = 0, A_WORDS
BASE_CIN, BASE_COUT = BASE_B + B_WORDS, BASE_B + B_WORDS + C_WORDS


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def pack2(lo, hi):
    return struct.unpack("<Q", struct.pack("<ff", f32(lo), f32(hi)))[0]


def run(cmd, stdin=None):
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True)


def vsm_of(fixture: pathlib.Path) -> str:
    r = run([str(BUILD / "bin/triton-opt"), str(fixture),
             "--tritongpu-to-mncore"])
    if r.returncode:
        raise RuntimeError("tritongpu-to-mncore: " + r.stderr[:600])
    r2 = run([str(LLVM / "mlir-opt"), "--convert-mncore-to-llvm"], r.stdout)
    if r2.returncode:
        raise RuntimeError("convert-mncore-to-llvm: " + r2.stderr[:600])
    r3 = run([str(LLVM / "mlir-translate"), "--mlir-to-llvmir"], r2.stdout)
    if r3.returncode:
        raise RuntimeError("mlir-translate: " + r3.stderr[:600])
    with tempfile.NamedTemporaryFile("w", suffix=".ll", delete=False) as f:
        f.write(r3.stdout)
        p = f.name
    r4 = run([str(LLVM / "llc"), "-mtriple=mncore2", p, "-o", "-"])
    if r4.returncode:
        raise RuntimeError("llc: " + r4.stderr[:600])
    lines, in_sidecar = [], False
    for line in r4.stdout.splitlines():
        s = line.strip()
        if not s or s.startswith("."):
            continue
        if s.startswith("#"):
            if "@mncore.fixup.begin" in s:
                in_sidecar = True
            if in_sidecar or "@mncore.fixup" in s or "@mncore.kernel" in s:
                lines.append(s)
            if "@mncore.fixup.end" in s:
                in_sidecar = False
            continue
        if s.endswith(":") or s.split()[0].endswith(":"):
            continue
        lines.append(s)
    return "\n".join(lines) + "\n"


def dset_dram(addr: int, value: int) -> str:
    spec = 9 | (addr << 4) | (1 << 21)
    return f"d set $d{addr} 1 {value:016x}"


def main() -> int:
    fixture = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 \
        else HERE / "t12_sliced_gemm.mlir"

    # C input: plane-major, block n's plane pl at longword n*256 + pl*64.
    # The values are ordinary floats, and the check is on the raw longword, so
    # an accidental arithmetic pass would also fail.
    cin = [pack2(1.0 + j, -2.0 - j) for j in range(C_WORDS)]
    preload = [dset_dram(BASE_CIN + j, v) for j, v in enumerate(cin)]

    vsm = vsm_of(fixture)
    bindings = {0: BASE_A * 8, 1: BASE_B * 8, 2: BASE_COUT * 8,
                3: BASE_CIN * 8}
    try:
        vsm, applied = mncore_fixup.relocate(vsm, bindings)
    except mncore_fixup.RelocError as e:
        print(f"E2E DOT FAIL: relocation: {e}\n{vsm}", file=sys.stderr)
        return 1
    # A 32 + B 32 + C read 16 + writeback 16, each a plane operand.
    if len(applied) != 8 * KB + 2 * KB * NB + 4 * NB + 4 * NB:
        print(f"E2E DOT FAIL: expected {8*KB + 2*KB*NB + 8*NB} relocated "
              f"operands, got {len(applied)}", file=sys.stderr)
        return 1

    body = "\n".join(preload) + "\n" + vsm + \
        f"d get $d{BASE_COUT} {C_WORDS}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".vsm", delete=False) as f:
        f.write(body)
        local = f.name
    stage = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", HOST, "cat > /tmp/mncore_e2e_dot.vsm"],
        stdin=open(local, "rb"))
    if stage.returncode:
        print("E2E DOT FAIL: staging", file=sys.stderr)
        return 1
    r = run(["ssh", "-o", "BatchMode=yes", HOST,
             f"{SDE} -hsw -- {ASM} --output-file /tmp/mncore_e2e_dot.asm "
             f"/tmp/mncore_e2e_dot.vsm"])
    if r.returncode or "racy" in r.stderr or "Error" in r.stderr:
        print("E2E DOT FAIL: assemble3\n" + (r.stdout + r.stderr)[:800],
              file=sys.stderr)
        return 1
    r = run(["ssh", "-o", "BatchMode=yes", HOST,
             f"{SDE} -hsw -- {EMU} -i /tmp/mncore_e2e_dot.asm "
             f"-d /tmp/mncore_e2e_dot.dmp"])
    if r.returncode:
        print("E2E DOT FAIL: gpfn3\n" + (r.stdout + r.stderr)[:800],
              file=sys.stderr)
        return 1

    dump = run(["ssh", "-o", "BatchMode=yes", HOST,
                "cat /tmp/mncore_e2e_dot.dmp"]).stdout
    groups = {}
    for m in re.finditer(
            r"DEBUG-DRAM\((n\d+),(\d+)\):.*?v:(0x[0-9a-fA-F]+)\) "
            r"#d get \$d" + str(BASE_COUT) + r" " + str(C_WORDS), dump):
        groups.setdefault(m.group(1), []).append(int(m.group(3), 16))
    if len(groups) != 4 or any(len(v) != C_WORDS for v in groups.values()):
        print(f"E2E DOT FAIL: expected 4 groups of {C_WORDS} values, got "
              f"{ {k: len(v) for k, v in groups.items()} }", file=sys.stderr)
        return 1
    for g, vals in sorted(groups.items()):
        bad = next((i for i, (a, b) in enumerate(zip(vals, cin)) if a != b),
                   None)
        if bad is not None:
            print(f"E2E DOT FAIL: group {g} differs at longword {bad}: "
                  f"{vals[bad]:#018x} vs {cin[bad]:#018x}", file=sys.stderr)
            return 1
    print(f"E2E DOT OK: {NB} N blocks moved C through "
          f"DMA -> slab -> gemm.tile -> slab -> DMA unchanged "
          f"({4*C_WORDS} DRAM longwords, 4 emulator groups). Plumbing only: "
          f"no product is checked (G2 open).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
