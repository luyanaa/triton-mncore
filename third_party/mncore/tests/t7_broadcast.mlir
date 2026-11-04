// Broadcast linear layout: lane and warp bases are zero, so every PE sees the
// same 4 f32 (l1bmp).

#bc = #ttg.linear<{register = [[1], [2]], lane = [[0], [0], [0], [0], [0]], warp = [[0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @bcast() {
    %m = ttg.local_alloc : () -> !ttg.memdesc<4xf32, #shared, #ttg.shared_memory, mutable>
    %v = ttg.local_load %m : !ttg.memdesc<4xf32, #shared, #ttg.shared_memory, mutable> -> tensor<4xf32, #bc>
    %r = arith.addf %v, %v : tensor<4xf32, #bc>
    tt.return
  }
}
