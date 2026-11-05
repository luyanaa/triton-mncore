// T9: a non-f32 dot must cite the MAU precision it would need.
//
// Only the f32 fmfma form (SDM 3.6.9.4) is defined here: an 8x4-by-4 block
// with K=4 and an f32 accumulation. f16 would need hmfma (3.6.9.6, a 16x16
// face with single-precision accumulation) and f64 the dmfma u|d row
// half-select (3.6.9.2, a 4x4 face where one instruction produces only two of
// the four rows). Neither is wired, so the rejection must name them rather
// than let the shape or layout checks produce a misleading message.

#blocked = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @dot16(%a: tensor<32x32xf16, #dotA>,
                 %b: tensor<32x32xf16, #dotB>,
                 %c: tensor<32x32xf16, #blocked>) {
    %d = tt.dot %a, %b, %c : tensor<32x32xf16, #dotA> * tensor<32x32xf16, #dotB> -> tensor<32x32xf16, #blocked>
    tt.return
  }
}
