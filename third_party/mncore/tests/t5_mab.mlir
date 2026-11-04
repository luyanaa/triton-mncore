// MAB-aware layout (SDM 3.6.8.8): the 4 PE values are repeated across all 16
// MABs. Linear layout: register = 4 f32 per PE, lane bits 0..1 = p, lane bits
// 2..4 = replicated, warp = replicated.

#mab = #ttg.linear<{register = [[4], [8]], lane = [[1], [2], [0], [0], [0]], warp = [[0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @mab() {
    %m = ttg.local_alloc : () -> !ttg.memdesc<16xf32, #shared, #ttg.shared_memory, mutable>
    %v = ttg.local_load %m : !ttg.memdesc<16xf32, #shared, #ttg.shared_memory, mutable> -> tensor<16xf32, #mab>
    %r = arith.addf %v, %v : tensor<16xf32, #mab>
    tt.return
  }
}
