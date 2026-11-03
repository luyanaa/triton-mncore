// T4/T5: distributed local_alloc -> local_load -> addf -> local_store.
// The 256 f32 tensor covers all 64 PEs; both directions use the two-longword
// plane split (low plane at L1BM 0, high plane at L1BM 64).

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @roundtrip() {
    %m = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared, #ttg.shared_memory, mutable>
    %v = ttg.local_load %m : !ttg.memdesc<256xf32, #shared, #ttg.shared_memory, mutable> -> tensor<256xf32, #blocked>
    %r = arith.addf %v, %v : tensor<256xf32, #blocked>
    ttg.local_store %r, %m : tensor<256xf32, #blocked> -> !ttg.memdesc<256xf32, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}
