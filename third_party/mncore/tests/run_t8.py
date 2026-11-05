#!/usr/bin/env python3
"""T8: the MAB and MAB4 local_store rejections are part of the contract.

Both are fail-closed by construction, not "not implemented yet":

  * MAB  — `l1bmm@<mabadr>` needs a physical MAB selector (0..15) that a
           replicated Triton layout does not determine, and one transfer
           writes a single PE's double-longword operand.
  * MAB4 — `l1bmm4@<k>` writes one of four groups, the groups hold different
           data, and `mncore.vec.store` carries a single value token.

A future change that silently picks a selector (or drops the group split)
breaks the physical model, so it must fail this test.

Usage: PYTHONPATH=python python run_t8.py
"""

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
TRITON_OPT = HERE.parents[2].parent / "triton-mncore-build" / "bin" / "triton-opt"

CASES = [
    ("t8_mab_store_reject.mlir",
     "MAB store is not derivable from a Triton layout",
     "physical MAB selector"),
    ("t8_mab4_store_reject.mlir",
     "MAB4 store is not expressible",
     "single value token"),
]


def main() -> int:
    failures = []
    for name, *needles in CASES:
        r = subprocess.run(
            [str(TRITON_OPT), str(HERE / name), "--tritongpu-to-mncore"],
            capture_output=True, text=True)
        out = r.stdout + r.stderr
        if r.returncode == 0:
            failures.append(f"{name}: accepted, but the store must be rejected")
            continue
        missing = [n for n in needles if n not in out]
        if missing:
            failures.append(f"{name}: rejection lacks {missing}: {out[:400]}")
            continue
        print(f"PASS  {name} rejected: {needles[0]}")

    if failures:
        for f in failures:
            print("T8 FAIL: " + f, file=sys.stderr)
        return 1
    print("T8 OK: MAB/MAB4 local_store rejected with the physical reason")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
