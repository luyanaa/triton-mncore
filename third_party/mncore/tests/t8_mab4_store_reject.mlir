// A MAB4-layout local_store must be rejected. `l1bmm4@<k>` writes exactly one
// of the four MAB groups, while the 4 groups hold *different* data (lane bit 4
// and the warp bit select the group). `mncore.vec.store` carries a single
// value token, so there is no way to express the four per-group sources.

#mab4 = #ttg.linear<{register = [[16], [32]], lane = [[1], [2], [0], [0], [4]], warp = [[8]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @mab4_store() {
    %m = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #ttg.shared_memory, mutable>
    %v = ttg.local_load %m : !ttg.memdesc<64xf32, #shared, #ttg.shared_memory, mutable> -> tensor<64xf32, #mab4>
    %r = arith.addf %v, %v : tensor<64xf32, #mab4>
    ttg.local_store %r, %m : tensor<64xf32, #mab4> -> !ttg.memdesc<64xf32, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}
