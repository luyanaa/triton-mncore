// T9: the tt.dot contract is fail-closed.
//
// The MN-Core MAU has no matrix-matrix instruction (SDM 3.6.9.1). One f32
// fmfma computes, per MAB, dst[i] = sum(j=0..3) A[i][2j]*x[j] + y[i] with
// A an 8x8 logical face (even columns only), x = 4 f32 at word width and
// y/dst = 8 f32 at longword width. A GEMM must be built from mwrite + one
// fmfma per 4 columns of B, chaining K through y.
//
// This is the TTGIR a real `tl.dot` (32x32x32, num_warps=2) produces: the
// blocked parent gives every thread a 4x4 output tile, and the dot-operand
// layout replicates the whole K dimension into registers. That placement is
// not the MAU's, so the pass must not lower it -- and the diagnostic must say
// which clause of the contract is missing rather than "unsupported op".

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @dot32(%a: tensor<32x32xf32, #dotA>,
                 %b: tensor<32x32xf32, #dotB>,
                 %c: tensor<32x32xf32, #blocked>) {
    %d = tt.dot %a, %b, %c : tensor<32x32xf32, #dotA> * tensor<32x32xf32, #dotB> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}
