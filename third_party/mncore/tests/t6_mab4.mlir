// MAB4-aware layout (SDM 3.6.8.14): lane bits 0..1 = p, bits 2..3 replicated,
// bit 4 selects the 4-MAB group, warp selects the upper group bit.

#mab4 = #ttg.linear<{register = [[16], [32]], lane = [[1], [2], [0], [0], [4]], warp = [[8]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @mab4() {
    %m = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #ttg.shared_memory, mutable>
    %v = ttg.local_load %m : !ttg.memdesc<64xf32, #shared, #ttg.shared_memory, mutable> -> tensor<64xf32, #mab4>
    %r = arith.addf %v, %v : tensor<64xf32, #mab4>
    tt.return
  }
}
