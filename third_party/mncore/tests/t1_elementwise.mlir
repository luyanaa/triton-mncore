// T1: minimal elementwise TTGIR. The mncore layout gives each PE four
// contiguous f32 (one 2-longword vector), order [0].
// RUN (manual, after the plugin is built):
//   triton-opt %s --tritongpu-to-mncore | mlir-opt --convert-mncore-to-llvm

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>

module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @add(%a: tensor<256xf32, #blocked>, %b: tensor<256xf32, #blocked>) {
    %0 = arith.addf %a, %b : tensor<256xf32, #blocked>
    %1 = arith.mulf %0, %a : tensor<256xf32, #blocked>
    tt.return
  }
}
