// T3: distributed tensor. 256 f32 elements over 64 PEs = 4 f32 (one
// 2-longword vector) per PE. l1bmd has no double-longword form, so the pass
// emits two longword planes (low at L1BM 0, high at L1BM 64).

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @dist() {
    %m = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared, #ttg.shared_memory, mutable>
    %v = ttg.local_load %m : !ttg.memdesc<256xf32, #shared, #ttg.shared_memory, mutable> -> tensor<256xf32, #blocked>
    %r = arith.addf %v, %v : tensor<256xf32, #blocked>
    tt.return
  }
}
