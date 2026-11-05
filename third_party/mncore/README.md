# MN-Core backend for Triton (skeleton)

This directory is the Triton-side adapter for the `mncore` dialect and the
MN-Core 2 LLVM backend that live in `/Users/yanlu/Documents/llvm-mncore`.

Status (2026-11-04): **M1 elementwise path closed on the emulator, with
relocatable buffer arguments, and the whole suite behind one entry point**
(`tests/run_all.py`; 13 stages, `--remote` adds the emulator E2E). The
Python stage table, the two C++ passes (`tritongpu-to-mncore`, and the
`convert-mncore-to-llvm` wrapper), and the LLVM backend are wired; a real
`@triton.jit` vector-add kernel runs through `assemble3` + `gpfn3` and matches
the CPU golden (`tests/run_e2e_add.py`). `tl.dot` reaches TTGIR (T9) and is
fail-closed at the MAU contract; the GEMM lowering is pending the MAU matrix
numeric golden.

Run everything (after rebuilding `libtriton.so` into `python/triton/_C/`):

```sh
cd third_party/mncore/tests
PYTHONPATH=../../../../triton-mncore/python python run_all.py [--remote]
```

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
`tests/run_all.py` is the single entry point for the whole suite (T0 plugin
import, T1..T7 layout/mode fixtures, T8 the MAB/MAB4 store rejections, R the
relocation contract, F the frontend regression; `--remote` adds the emulator
E2E).

## Known risks

* Triton 3.9's MLIR/LLVM API surface against LLVM 24 is untested.
* Triton's TTGIR layout heuristics assume a GPU warp; the `mnir` pass must not
  rely on them. The current `make_ttgir` uses a placeholder target string and
  `warp_size=32`.
* Buffer arguments are relocated, not folded. `.vsm` addresses are static, so
  every DRAM operand derived from a kernel pointer is emitted as placeholder 0
  plus a `# @mncore.fixup <site>` comment, with the record in the
  `@mncore.fixup.begin` sidecar block at end of file (kernel ABI v1 §13).
  `tests/mncore_fixup.py` is the reference consumer: `relocate(vsm, {arg:
  base_bytes})` binds arguments, checks unit alignment and `width_bits`, and
  strips the comments for `assemble3`. `tests/run_reloc.py` covers the positive
  path and the fail-closed rejections; `run_e2e_add.py` runs the whole pipeline
  through it.
* `ttg.local_load` is mode-driven (SDM 3.6.8): Broadcast -> `l1bmp`, MAB ->
  `l1bmm`, MAB4 -> `l1bmm4` (all single double-longword transfers); a full
  64-PE distribution -> two `l1bmd` transfers, one longword plane each (l1bmd
  has no double-longword form). The pass writes the slot's longwords at
  `W/2` and `W/2+4`; the L1BM layout must place the low plane at 64
  longwords and the high plane immediately after it.
* `ttg.local_store` distributed is numerically closed (measured on gpfn3).
  One `l1bmd` collect already writes **all 64 PE lanes**: SDM §3.6.8.21 has
  `data[(mab+mabdiff)%16*4 + pe] = PE[mab][pe]`, so `mabdiff` is a **MAB
  rotation (a permutation), not a mask**; the "one collect per MAB / 16
  rotated collects" plan was wrong. Only the 64-longword destination must be
  64-aligned (assembler rule).
* The **L1BM → L2BM → DRAM** half is where the E2E mismatch actually lived.
  `l2bmd` (collect form, §3.6.7.8) composes L2BM out of 8-longword chunks
  taken from *each of the eight L1Bs in turn*. A Triton "distributed" tile is
  replicated across L1Bs, so the collect interleaved eight identical chunks
  and destroyed the plane. Writeback now uses `l2bm@<l1b>` (single-L1B point
  transfer, §3.6.7.5) on L1B 0: one contiguous 64-longword stream per plane.
  `l2bmd` remains the correct primitive only once L1Bs hold disjoint data.
* MAB/MAB4 **stores** are rejected by construction, and the rejection is
  pinned by `tests/run_t8.py` (`t8_mab_store_reject.mlir`,
  `t8_mab4_store_reject.mlir`): `l1bmm@<mabadr>` needs a physical MAB
  selector (0..15) that a replicated Triton layout does not determine, and
  `l1bmm4@<k>` writes one of four groups whose contents differ, while
  `mncore.vec.store` carries a single value token. Use an explicit
  `mncore.vec.store` with a chosen `set`, or split the tensor per group.
* `tt.dot` **reaches TTGIR and is gated**, not lowered (`tests/run_t9.py`).
  The frontend needs the dot policy fields on `MncoreOptions` and a
  `min_dot_size` hook on `get_codegen_implementation`; a real 32x32x32
  `tl.dot` then produces `#blocked` (4x4 C tile per thread) with
  `#ttg.dot_op<parent=#blocked>` operands. `tritongpu-to-mncore` rejects it
  with the failing clause of the MAU contract (precise e.g. "is f32 only",
  "does not tile the MAU", or the pending GEMM sequence). The ISA has no
  matrix-matrix instruction (SDM 3.6.9.1: `mfma`/`mmul`, one matrix operand
  each), and the blocked FMA dot-operand layout replicates the whole K
  dimension into registers while the MAU consumes K one element per PE.
  Nothing here is lowered until the MAU matrix primitives have a numeric
  golden on gpfn3; see `mncore_intrinsics_design.md` 22 for the geometry and
  the open width question.
