// T2: local_alloc -> local_load, broadcast distribution. Each thread owns the
// whole 4xf32 vector, so every PE sees the same value and the mode is
// Broadcast (l1bmp), not a per-PE distribution.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @bcast() {
    %m = ttg.local_alloc : () -> !ttg.memdesc<4xf32, #shared, #ttg.shared_memory, mutable>
    %v = ttg.local_load %m : !ttg.memdesc<4xf32, #shared, #ttg.shared_memory, mutable> -> tensor<4xf32, #blocked>
    %r = arith.addf %v, %v : tensor<4xf32, #blocked>
    tt.return
  }
}
