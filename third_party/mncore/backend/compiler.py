from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, fields

from triton._C.libtriton import ir, llvm, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language

# The MN-Core 2 target name registered in the LLVM backend
# (llvm/lib/Target/MNCore/TargetInfo/MNCoreTargetInfo.cpp).
MN_CORE_TRIPLE = "mncore2"


@dataclass(frozen=True)
class MncoreOptions:
    """Options accepted by the mncore backend.

    `num_warps` is kept only for Triton API compatibility: MN-Core maps a
    program to PE lanes at compile time, so there is no runtime warp count.
    """

    num_warps: int = 1
    num_ctas: int = 1
    num_stages: int = 1
    debug: bool = False
    backend_name: str = "mncore"
    arch: str = "mncore2"
    # The MAU is a plain IEEE f32 matrix unit: there is no tensor-core
    # reduced-precision mode, so `ieee` is the only input precision. The
    # frontend reads these fields from `tl.dot`'s semantic routine.
    sanitize_overflow: bool = True
    allowed_dot_input_precisions: tuple = ("ieee",)
    default_dot_input_precision: str = "ieee"
    max_num_imprecise_acc_default: int = 0
    deprecated_fp8_dot_operand_dtypes: tuple = ()

    def hash(self) -> str:
        key = "_".join(f"{f.name}-{getattr(self, f.name)}" for f in fields(self))
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _mncore_module():
    """Return the plugin submodule, or None when it has not been built.

    The plugin is registered by main.cc as a top-level `libtriton.mncore`
    submodule (see INIT_BACKEND in python/src/main.cc), not under `passes`.
    """
    try:
        from triton._C.libtriton import mncore
    except ImportError:
        return None
    return mncore


class MncoreBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "mncore"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def hash(self) -> str:
        return f"mncore-{MN_CORE_TRIPLE}"

    def parse_options(self, opts) -> MncoreOptions:
        allowed = {f.name for f in fields(MncoreOptions)}
        kwargs = {k: v for k, v in opts.items() if k in allowed}
        options = MncoreOptions(**kwargs)
        if options.num_warps not in (1, 2) or options.num_ctas != 1:
            raise ValueError(
                "MN-Core maps threads to 64 PE lanes: num_warps must be 1 "
                "(32 lanes, broadcast/partial) or 2 (64 lanes, distributed); "
                "num_ctas must be 1")
        if options.num_stages != 1:
            raise ValueError(
                "MN-Core double buffering is not wired yet; num_stages must be 1")
        return options

    def load_dialects(self, ctx):
        module = _mncore_module()
        if module is not None:
            module.load_dialects(ctx)

    def get_module_map(self):
        return {}

    def get_codegen_implementation(self, options):
        # No libdevice/extra codegen functions yet. The frontend still needs
        # a `min_dot_size` hook to accept `tl.dot` at all; (1, 1, 1) defers
        # the shape contract to `tritongpu-to-mncore`, which rejects every
        # shape it cannot map to the 4-wide MAU exactly.
        return {"min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1)}

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def add_stages(self, stages, options, language):
        if language != Language.TRITON:
            raise NotImplementedError(
                "the mncore backend currently accepts Triton kernels only")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["mnir"] = lambda src, metadata: self.make_mnir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["vsm"] = lambda src, metadata: self.make_vsm(src, metadata, options)

    def make_ttir(self, mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        # The MN-Core backend has no control flow: unroll at TTIR level so the
        # later stages see straight-line code only.
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    def make_ttgir(self, mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # Placeholder layout pass. The mncore layout mapping is defined in the
        # `tritongpu-to-mncore` conversion; the GPU warp size here is only a
        # value the pass requires, not a hardware property.
        passes.ttir.add_convert_to_ttgpuir(
            pm, f"mncore:{MN_CORE_TRIPLE}", options.num_warps, 32, 1)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod, "make_ttgir")
        return mod

    def make_mnir(self, mod, metadata, options):
        module = _mncore_module()
        if module is None:
            raise RuntimeError(
                "the mncore plugin is not built: third_party/mncore has no "
                "tritongpu-to-mncore pass yet. See third_party/mncore/README.md.")
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        module.add_tritongpu_to_mncore(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod, "make_mnir")
        return mod

    def make_llir(self, mod, metadata, options):
        module = _mncore_module()
        if module is None:
            raise RuntimeError(
                "the mncore plugin is not built: third_party/mncore has no "
                "convert-mncore-to-llvm wrapper. See third_party/mncore/README.md.")
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        module.add_convert_mncore_to_llvm(pm)
        pm.run(mod, "make_llir")

        # Triton's built-in init only registers NVPTX/AMDGPU; the plugin has
        # already initialized the MNCore target.
        if hasattr(module, "init_targets"):
            module.init_targets()
        else:
            llvm.init_targets()

        ctx = llvm.context()
        llvm_mod = llvm.to_module(mod, ctx)
        llvm.attach_datalayout(llvm_mod, MN_CORE_TRIPLE, "", "", [])
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
        metadata["name"] = mod.get_str_attr("tt.name") or "kernel"
        ret = str(llvm_mod)
        del llvm_mod
        del ctx
        return ret

    def make_vsm(self, src, metadata, options):
        asm = llvm.translate_to_asm(
            src,
            MN_CORE_TRIPLE,
            "",  # CPU
            "",  # features
            [],  # LLVM flags
            enable_fp_fusion=False,
            isObject=False,
            canonicalizeGEP=False,
        )
        names = re.findall(r"^\s*\.globl\s+([A-Za-z_][A-Za-z0-9_]*)",
                           asm, flags=re.MULTILINE)
        metadata["name"] = names[0] if len(names) == 1 else metadata.get("name", "kernel")
        metadata["assembly_format"] = "mncore-vsm"
        return asm.encode("utf-8")
