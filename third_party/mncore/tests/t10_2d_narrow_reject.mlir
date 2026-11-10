// T10 negative: a 2-D tile whose per-PE width is not a whole longword pair.
//
// A PE register slot is two longwords (4 f32). sizePerThread [1, 2] gives two
// register bits' worth of *nothing*: 2 f32 per thread is half a slot, so the
// tile would occupy only half of each PE's vector and the other half would be
// stale on the way into a `vec.*` expression. The pass must reject it rather
// than round it up.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>

module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @dist2d_narrow() {
    %m = ttg.local_alloc : () -> !ttg.memdesc<8x16xf32, #shared, #ttg.shared_memory, mutable>
    %v = ttg.local_load %m : !ttg.memdesc<8x16xf32, #shared, #ttg.shared_memory, mutable> -> tensor<8x16xf32, #blocked>
    %r = arith.addf %v, %v : tensor<8x16xf32, #blocked>
    tt.return
  }
}
