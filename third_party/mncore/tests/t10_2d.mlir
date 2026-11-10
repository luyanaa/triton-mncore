// T10: a rank-2 (2-D) distributed tile.
//
// The distribution criterion is rank-agnostic (design doc: the pass classifies
// by which lane bits are replicated, not by the tensor rank), so a 16x16 f32
// tile with 4 f32 per PE is Distributed exactly as its 256-element 1-D
// flattening is. The contract that makes this sound is that the logical
// row-major order of the tile IS the physical L1BM/PE order -- the DMA path
// below emits the same two 64-longword planes it emits for tensor<256xf32>.
//
// 16x16 = 256 elements over 64 PEs = 4 f32 per PE. sizePerThread [1, 4] gives
// exactly two register bits, which is what `lanes = 4` for f32 requires.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>

module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @dist2d() {
    %m = ttg.local_alloc : () -> !ttg.memdesc<16x16xf32, #shared, #ttg.shared_memory, mutable>
    %v = ttg.local_load %m : !ttg.memdesc<16x16xf32, #shared, #ttg.shared_memory, mutable> -> tensor<16x16xf32, #blocked>
    %r = arith.addf %v, %v : tensor<16x16xf32, #blocked>
    tt.return
  }
}
