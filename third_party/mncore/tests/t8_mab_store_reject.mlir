// A MAB-layout local_store must be rejected, and the rejection is part of the
// contract: `l1bmm@<mabadr>` needs a physical MAB selector (0..15) that a
// Triton linear layout does not determine, and the transfer footprint is one
// PE's double-longword operand, not the whole 4-PE MAB lane. Do not let a
// future change silently pick a selector.

#mab = #ttg.linear<{register = [[4], [8]], lane = [[1], [2], [0], [0], [0]], warp = [[0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @mab_store() {
    %m = ttg.local_alloc : () -> !ttg.memdesc<16xf32, #shared, #ttg.shared_memory, mutable>
    %v = ttg.local_load %m : !ttg.memdesc<16xf32, #shared, #ttg.shared_memory, mutable> -> tensor<16xf32, #mab>
    %r = arith.addf %v, %v : tensor<16xf32, #mab>
    ttg.local_store %r, %m : tensor<16xf32, #mab> -> !ttg.memdesc<16xf32, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}
