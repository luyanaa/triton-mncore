// T9: a dot whose shape does not tile the MAU face is rejected at the shape
// clause, before any layout question.
//
// The f32 matrix face is 8 logical rows (M) shared by the MAB's 4 PEs, and one
// fmfma consumes 4 columns of B (N) and 4 of K. N=2 here cannot be covered by
// a single instruction's four cycle vectors, so the shape is not merely
// "unsupported" -- it does not tile.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @dot_bad_n(%a: tensor<8x4xf32, #dotA>,
                     %b: tensor<4x2xf32, #dotB>,
                     %c: tensor<8x2xf32, #blocked>) {
    %d = tt.dot %a, %b, %c : tensor<8x4xf32, #dotA> * tensor<4x2xf32, #dotB> -> tensor<8x2xf32, #blocked>
    tt.return
  }
}
