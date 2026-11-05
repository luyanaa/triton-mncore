#!/usr/bin/env python3
"""Run the whole mncore test suite in process order.

Stages:
  T0  plugin import / backend discovery      (run_t0.sh)
  T1  elementwise -> mncore.vec.*
  T2  broadcast  -> mncore.vec.load
  T3  distributed load -> l1bmd (mode 3)
  T4  distributed load+store round trip
  T5  MAB load   -> l1bmm  (mode 1)
  T6  MAB4 load  -> l1bmm4 (mode 2)
  T7  Broadcast  -> l1bmp  (mode 0)
  T8  MAB/MAB4 store non-derivability
  T9  tt.dot / GEMM contract
  R   relocation contract (§13.3/§13.5)      (run_reloc.py)
  F   Triton frontend regression             (run_frontend.py)

`run_e2e_add.py` needs ssh to the emulator host and a working assemble3/gpfn3,
so it stays opt-in (`--remote`).

Usage:
  PYTHONPATH=python python run_all.py [--remote]
"""

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable
BASE_ENV = {"PYTHONPATH": str(HERE.parents[2] / "python")}

STAGES = [
    ("T1", ["run_t1.py", "t1_elementwise.mlir"]),
    ("T2", ["run_t2.py", "t2_broadcast.mlir"]),
    ("T3", ["run_t3.py", "t3_distributed.mlir"]),
    ("T4", ["run_t4.py", "t4_roundtrip.mlir"]),
    ("T5", ["run_t5.py", "t5_mab.mlir"]),
    ("T6", ["run_t6.py", "t6_mab4.mlir"]),
    ("T7", ["run_t7.py", "t7_broadcast.mlir"]),
    ("T8", ["run_t8.py"]),
    ("T9", ["run_t9.py"]),
    ("R", ["run_reloc.py"]),
    ("F", ["run_frontend.py"]),
]


def run(cmd, cwd, env):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          env=env)


def main() -> int:
    import os

    env = dict(os.environ)
    env.update(BASE_ENV)
    # run_t0.sh defaults to `python` on PATH; pin it to the interpreter that
    # is already running this suite (the venv that has triton's deps).
    env["PYTHON"] = PY

    failures = []
    # T0 doubles as the libtriton.so install step, so it must run first.
    r = run(["bash", "run_t0.sh"], HERE, env)
    if r.returncode != 0:
        failures.append(("T0", r.stdout + r.stderr))
        # Without the plugin nothing else can run.
        for label, _ in STAGES:
            failures.append((label, "skipped: T0 failed"))
    else:
        print(f"PASS  T0  {r.stdout.strip().splitlines()[-1]}")
        for label, argv in STAGES:
            r = run([PY] + argv, HERE, env)
            # The T* runners print the module before the verdict; keep only
            # the verdict line in the summary.
            verdict = ""
            for line in (r.stdout + r.stderr).strip().splitlines()[::-1]:
                if line.startswith(label) and ("OK" in line or "FAIL" in line):
                    verdict = line
                    break
            if r.returncode != 0:
                failures.append((label, (r.stdout + r.stderr)[-1500:]))
                print(f"FAIL  {label}")
            else:
                print(f"PASS  {label}  {verdict}")

    if "--remote" in sys.argv:
        r = run([PY, "run_e2e_add.py"], HERE, env)
        if r.returncode != 0:
            failures.append(("E2E", (r.stdout + r.stderr)[-1500:]))
            print("FAIL  E2E")
        else:
            print("PASS  E2E  " + r.stdout.strip())

    if failures:
        for label, out in failures:
            print(f"\n===== {label} =====\n{out}", file=sys.stderr)
        print(f"\nSUITE FAILED: {len(failures)} stage(s)", file=sys.stderr)
        return 1
    print("\nMNCore SUITE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
