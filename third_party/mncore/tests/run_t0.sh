#!/usr/bin/env bash
# T0: plugin import and backend discovery. Run after building libtriton.so.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
build="${TRITON_BUILD_DIR:-$repo/../triton-mncore-build}"

mkdir -p "$repo/python/triton/_C"
cp "$build/libtriton.so" "$repo/python/triton/_C/libtriton.so"

PYTHONPATH="$repo/python" TRITON_BACKENDS_IN_TREE=1 "${PYTHON:-python}" - <<'PY'
import triton
from triton.backends import backends
assert "mncore" in backends, sorted(backends)
from triton.backends.mncore.compiler import MncoreBackend
from triton.backends.compiler import GPUTarget
b = MncoreBackend(GPUTarget("mncore", "mncore2", 1))
print("T0 OK:", sorted(backends), b.hash())
PY
