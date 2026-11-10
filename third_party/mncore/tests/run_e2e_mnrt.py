#!/usr/bin/env python3
"""P3 runtime path: the same two-tile kernel, executed by libmnrt.

`run_e2e_add2d.py` drives the emulator the way a build script would: it stages
a .vsm, runs assemble3, runs gpfn3, and reads the dump. That proves the
*compiler*, but it bypasses the runtime the device path is built on.

This script executes the identical program (same preload, same relocated
kernel, same `d get` readback -- `run_e2e_add2d.full_body()`) through libmnrt's
stable C ABI:

    mnrt_load_program() -> mnrt_enqueue() -> mnrt_wait()

so the buffer arguments are bound by the same relocation contract the loader
uses (kernel ABI §13.3), and the dump is read back through MNRT_DUMP. The
comparison is element for element against the same CPU golden.

Two properties are therefore checked at once:
  * the compiler's `.vsm` is a valid runtime image for mnrt's emulator
    provider (it parses, enqueues, and executes as one program), and
  * the numbers still match, i.e. routing through the runtime does not
    change the result.

The driver (`mnrt_run.c`, this directory) is compiled on the host against the
checkout's `libmnrt.a`; nothing is written into the libmnrt tree.

Environment: MNRT_HOST (default suzhou), MNRT_ROOT (default ~/libmnrt),
MNRT_SDE / MNRT_ASM / MNRT_EMU as in libmnrt's README.
"""

import os
import pathlib
import subprocess
import sys
import tempfile

import run_e2e_add2d as base

HERE = pathlib.Path(__file__).resolve().parent
HOST = os.environ.get("MNRT_HOST", base.HOST)
ROOT = os.environ.get("MNRT_ROOT", "~/libmnrt")
SDE = os.environ.get("SDE",
                     "/opt/sde-external-10.13.1-2026-07-28-lin/sde64")
ASM = os.environ.get("ASSEMBLE3",
                     "/opt/pfn/pfcomp/codegen/build/tools/assemble3")
EMU = os.environ.get("GPFN3",
                     "/opt/pfn/pfcomp/codegen/build/tools/gpfn3_package_main")
WORK = "/tmp/mncore_mnrt_e2e"


def ssh(cmd: str, stdin=None):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, cmd],
                          stdin=stdin, capture_output=True, text=True)


def main() -> int:
    try:
        body, golden = base.full_body()
    except base.mncore_fixup.RelocError as e:
        print(f"E2E MNRT FAIL: relocation: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"E2E MNRT FAIL: {e}", file=sys.stderr)
        return 1

    with tempfile.NamedTemporaryFile("w", suffix=".vsm", delete=False) as f:
        f.write(body)
        local = f.name
    with open(local, "rb") as f:
        staged = ssh(f"mkdir -p {WORK} && cat > {WORK}/kernel.vsm", stdin=f)
    if staged.returncode != 0:
        print("E2E MNRT FAIL: staging\n" + staged.stderr[:400], file=sys.stderr)
        return 1
    with open(HERE / "mnrt_run.c", "rb") as f:
        drv = ssh(f"mkdir -p {WORK} && cat > {WORK}/mnrt_run.c", stdin=f)
    if drv.returncode != 0:
        print("E2E MNRT FAIL: staging driver\n" + drv.stderr[:400],
              file=sys.stderr)
        return 1

    build = ssh(
        f"cd {WORK} && cc -O2 -I {ROOT}/include mnrt_run.c {ROOT}/libmnrt.a "
        f"-o mnrt_run")
    if build.returncode != 0:
        print("E2E MNRT FAIL: driver build (is libmnrt built?)\n"
              + (build.stdout + build.stderr)[:600], file=sys.stderr)
        return 1

    env = (f"MNRT_ROOT={ROOT} MNRT_SDE={SDE} MNRT_ASM={ASM} MNRT_EMU={EMU} "
           f"MNRT_DUMP={WORK}/e2e.dmp")
    run = ssh(f"cd {WORK} && {env} ./mnrt_run kernel.vsm")
    if run.returncode != 0:
        print("E2E MNRT FAIL: runtime\n" + (run.stdout + run.stderr)[:600],
              file=sys.stderr)
        return 1
    if "MNRT OK" not in run.stdout:
        print("E2E MNRT FAIL: no completion line\n" + run.stdout[:400],
              file=sys.stderr)
        return 1

    dump = ssh(f"cat {WORK}/e2e.dmp")
    if dump.returncode != 0:
        print("E2E MNRT FAIL: dump readback\n" + dump.stderr[:400],
              file=sys.stderr)
        return 1
    for tile in base.OUTPUTS:
        groups = base.dump_groups(dump.stdout, base.BASE[tile],
                                  2 * base.PLANE_LONGWORDS)
        want = golden[tile]
        if len(groups) != 4 or any(len(v) != len(want) for v in groups.values()):
            print(f"E2E MNRT FAIL: {tile}: expected 4 groups of {len(want)}, "
                  f"got { {k: len(v) for k, v in groups.items()} }",
                  file=sys.stderr)
            return 1
        for g, vals in sorted(groups.items()):
            bad = next((i for i, (x, y) in enumerate(zip(vals, want))
                        if x != y), None)
            if bad is not None:
                print(f"E2E MNRT FAIL: {tile}/{g} differs at {bad}: "
                      f"{vals[bad]:#x} vs {want[bad]:#x}", file=sys.stderr)
                return 1
    print(f"E2E MNRT OK: program loaded and run by libmnrt "
          f"({run.stdout.strip()}); 2 tiles x 4 x {2 * base.PLANE_LONGWORDS} "
          f"DRAM longwords match the CPU golden element for element")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
