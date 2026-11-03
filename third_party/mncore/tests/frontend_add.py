import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.mncore.compiler import MncoreBackend
from triton._C.libtriton import ir


@triton.jit
def addk(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(out_ptr + offs, x + y)


def main():
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
        constexprs={"BLOCK": 256})
    meta = {}
    mod = src.make_ir(ctx, opts)
    mod = backend.make_ttir(mod, meta, opts)
    print("=== TTIR ===")
    print(str(mod))
    mod = backend.make_ttgir(mod, meta, opts)
    print("=== TTGIR ===")
    print(str(mod))


if __name__ == "__main__":
    main()
