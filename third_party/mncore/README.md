# MN-Core backend for Triton (skeleton)

This directory is the Triton-side adapter for the `mncore` dialect and the
MN-Core 2 LLVM backend that live in `/Users/yanlu/Documents/llvm-mncore`.

Status (2026-11-03): **skeleton**. The Python stage table and target plumbing
are in place; the two C++ passes (`tritongpu-to-mncore`, and the
`convert-mncore-to-llvm` wrapper) are not built yet, so `mnir`/`llir` raise a
clear error until the plugin is wired.

## Contract

```
TTIR -> TTGIR -> mncore dialect -> LLVM IR (llvm.mncore.*) -> .vsm
```

* `convert-mncore-to-llvm` comes from the LLVM/MLIR install:
  `lib/libMLIRMNCoreToLLVM.a` (`/Users/yanlu/Documents/llvm-mncore`).
* `.vsm` is produced in-process by `llvm.translate_to_asm(src, "mncore2", ...)`,
  which runs `MNCoreAsmPrinter` from the same LLVM build. No subprocess, no
  version skew.
* The runtime/launcher is owned by the device-integration work line. This
  backend only produces `.vsm` + metadata.

## Build

```sh
cmake -S llvm -B build-mlir -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_TARGETS_TO_BUILD= -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=MNCore
cmake --install build-mlir --prefix /opt/llvm-mncore

# in ../triton-mncore
export LLVM_SYSPATH=/opt/llvm-mncore
```

Enable the backend by adding `"mncore"` to
`BackendInstaller.copy(["nvidia", "amd"])` in `setup.py` (one line). That also
adds it to `TRITON_CODEGEN_BACKENDS`, which makes CMake descend into this
directory.

## Required C++ work

1. `lib/Conversion/TritonGPUToMNCore.cpp`
   * `tt.load`/`tt.store` -> `mncore.mv` + `mncore.l2b` + `mncore.l1b`
   * `tt.dot` -> `mncore.mat.write` / `mncore.mat.vec` / `mncore.mat.read`
   * elementwise -> `mncore.vec.*`
   * block/layout -> per-PE 2-longword slices
   * reject anything that cannot be made straight-line: the LLVM backend
     rejects `BR/BRCOND/BR_CC/BR_JT/BRIND`.
2. `triton_mncore.cc`
   * `mncore.load_dialects(ctx)` — register `MNCoreDialect`.
   * `mncore.add_tritongpu_to_mncore(pm)`.
   * `mncore.add_convert_mncore_to_llvm(pm)` — wraps
     `createConvertMNCoreToLLVMPass()`.
   * **Initialize the target**: Triton's built-in `llvm.init_targets()` only
     initializes NVPTX/AMDGPU. The plugin must call
     `LLVMInitializeMNCoreTargetInfo/Target/TargetMC/AsmPrinter`.
3. `CMakeLists.txt` — `add_triton_plugin(TritonMncore triton_mncore.cc
   LINK_LIBS PUBLIC MLIRMNCoreDialect MLIRMNCoreToLLVM)`.

## Layout derivation

The distribution mode is identified from the linear layout's **replicated
lane bits** (see `llvm/lib/Target/MNCore/docs/mncore_mlir_dialect.md` §8.3b):
all six = Broadcast (l1bmp), bits 2..5 = MAB (l1bmm), bits 2..3 = MAB4
(l1bmm4), none = Distributed (two-plane l1bmd). Element order (strided,
contiguous, plane-major) is not part of the criterion. Blocked layouts
express Distributed and (degenerate) MAB; MAB4/Broadcast need `#ttg.linear`.

`tests/run_frontend.py` drives the real Triton frontend (TTIR/TTGIR) and
checks both the legalizer diagnostic and the MAB-aware fixtures end-to-end.

## Known risks

* Triton 3.9's MLIR/LLVM API surface against LLVM 24 is untested.
* Triton's TTGIR layout heuristics assume a GPU warp; the `mnir` pass must not
  rely on them. The current `make_ttgir` uses a placeholder target string and
  `warp_size=32`.
* Buffer arguments need loader relocation: `.vsm` addresses are static. See
  `llvm/lib/Target/MNCore/docs/mncore_frontend_integration.md` §2.4.
* `ttg.local_load` is mode-driven (SDM 3.6.8): Broadcast -> `l1bmp`; a full
  64-PE distribution -> two `l1bmd` transfers, one longword plane each (l1bmd
  has no double-longword form). The pass writes the slot's longwords at
  `W/2` and `W/2+4`; the L1BM layout must place the low plane at 64
  longwords and the high plane immediately after it. `l1bmm`/`l1bmm4`
  require a future MAB-aware encoding (not derivable from Triton layouts).
* `ttg.local_store` distributed is **not numerically closed**: one `l1bmd`
  collect per plane captures only one MAB (4 lanes); a full 64-PE writeback
  needs 16 `mabdiff`-rotated collects (4-lane step) and the L1BM subset
  layout (SDM §3.6.8.21/§3.7). `run_e2e_add.py` uses varied inputs and fails
  the multiset check until this is implemented; constant inputs hide it. MAB/MAB4 stores are not derivable from Triton layouts.
