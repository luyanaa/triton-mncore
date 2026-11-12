// T12: the blocked GEMM -- a legal rank-2 f32 dot lowered to a grid of
// mncore.gemm.tile blocks.
//
// This is the TTGIR the mncore frontend produces for a 128x16x16 `tl.dot`
// (`num_warps = 2`, the same layout choices `fmaDotToLinearLayout` makes),
// with one change: the caller supplies the accumulator as a loaded tile
// instead of letting Triton synthesise `dense<0.0>`. The MAU has no zero-fill
// form in this path (mmul, matvec with y forced to 0, is not emitted), so a
// dot that starts from a compiler-made zero is rejected -- see
// mncore_intrinsics_design.md 22.34.
//
// The contract (22.34): M = 128 is one gemm.tile grid (16 MABs x 8 face rows),
// K and N are multiples of the 4 the MAU consumes, so KB = K/4 = 4 and
// NB = N/4 = 4 here. The four buffers are *pre-arranged*, not the logical
// tiles: the pass reads 2x numel(A) and 16x numel(B) f32 from the A/B base
// pointers (the face pads A to even columns, and l1bmd gives each of the 64
// PEs its own x), and numel(C) from the accumulator. The C operand is read and
// the tt.store's buffer is written -- they need not be the same argument.
//
// Structural only: G2 (fmfma numerics) is open, so nothing here says anything
// about the numbers.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#s2 = #ttg.slice<{dim = 1, parent = #blocked2}>
#s3 = #ttg.slice<{dim = 0, parent = #blocked3}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "mncore:mncore2", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gemm128(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>, %acc_ptr: !tt.ptr<f32>) {
    %k = arith.constant dense<16> : tensor<16x1xi32, #blocked1>
    %cst = arith.constant dense<16> : tensor<128x1xi32, #blocked1>
    %a = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #s2>
    %a0 = tt.expand_dims %a {axis = 1 : i32} : tensor<128xi32, #s2> -> tensor<128x1xi32, #blocked2>
    %a1 = ttg.convert_layout %a0 : tensor<128x1xi32, #blocked2> -> tensor<128x1xi32, #blocked1>
    %a2 = arith.muli %a1, %cst : tensor<128x1xi32, #blocked1>
    %a3 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    %a4 = tt.addptr %a3, %a2 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
    %a5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #s3>
    %a6 = tt.expand_dims %a5 {axis = 0 : i32} : tensor<16xi32, #s3> -> tensor<1x16xi32, #blocked3>
    %a7 = ttg.convert_layout %a6 : tensor<1x16xi32, #blocked3> -> tensor<1x16xi32, #blocked4>
    %a8 = tt.broadcast %a4 : tensor<128x1x!tt.ptr<f32>, #blocked1> -> tensor<128x16x!tt.ptr<f32>, #blocked1>
    %a9 = ttg.convert_layout %a8 : tensor<128x16x!tt.ptr<f32>, #blocked1> -> tensor<128x16x!tt.ptr<f32>, #blocked4>
    %a10 = tt.broadcast %a7 : tensor<1x16xi32, #blocked4> -> tensor<128x16xi32, #blocked4>
    %a11 = tt.addptr %a9, %a10 : tensor<128x16x!tt.ptr<f32>, #blocked4>, tensor<128x16xi32, #blocked4>
    %a12 = tt.load %a11 : tensor<128x16x!tt.ptr<f32>, #blocked4>
    %b0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #s2>
    %b1 = tt.expand_dims %b0 {axis = 1 : i32} : tensor<16xi32, #s2> -> tensor<16x1xi32, #blocked2>
    %b2 = ttg.convert_layout %b1 : tensor<16x1xi32, #blocked2> -> tensor<16x1xi32, #blocked1>
    %b3 = arith.muli %b2, %k : tensor<16x1xi32, #blocked1>
    %b4 = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked1>
    %b5 = tt.addptr %b4, %b3 : tensor<16x1x!tt.ptr<f32>, #blocked1>, tensor<16x1xi32, #blocked1>
    %b6 = tt.broadcast %b5 : tensor<16x1x!tt.ptr<f32>, #blocked1> -> tensor<16x16x!tt.ptr<f32>, #blocked1>
    %b7 = ttg.convert_layout %b6 : tensor<16x16x!tt.ptr<f32>, #blocked1> -> tensor<16x16x!tt.ptr<f32>, #blocked4>
    %b8 = tt.broadcast %a7 : tensor<1x16xi32, #blocked4> -> tensor<16x16xi32, #blocked4>
    %b9 = tt.addptr %b7, %b8 : tensor<16x16x!tt.ptr<f32>, #blocked4>, tensor<16x16xi32, #blocked4>
    %b10 = tt.load %b9 : tensor<16x16x!tt.ptr<f32>, #blocked4>
    %acc0 = tt.splat %acc_ptr : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    %acc1 = tt.addptr %acc0, %a2 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
    %acc2 = tt.broadcast %acc1 : tensor<128x1x!tt.ptr<f32>, #blocked1> -> tensor<128x16x!tt.ptr<f32>, #blocked1>
    %acc3 = ttg.convert_layout %acc2 : tensor<128x16x!tt.ptr<f32>, #blocked1> -> tensor<128x16x!tt.ptr<f32>, #blocked>
    %acc4 = ttg.convert_layout %a10 : tensor<128x16xi32, #blocked4> -> tensor<128x16xi32, #blocked>
    %acc5 = tt.addptr %acc3, %acc4 : tensor<128x16x!tt.ptr<f32>, #blocked>, tensor<128x16xi32, #blocked>
    %c = tt.load %acc5 : tensor<128x16x!tt.ptr<f32>, #blocked>
    %a13 = ttg.convert_layout %a12 : tensor<128x16xf32, #blocked4> -> tensor<128x16xf32, #dotA>
    %b11 = ttg.convert_layout %b10 : tensor<16x16xf32, #blocked4> -> tensor<16x16xf32, #dotB>
    %d = tt.dot %a13, %b11, %c : tensor<128x16xf32, #dotA> * tensor<16x16xf32, #dotB> -> tensor<128x16xf32, #blocked>
    %d0 = ttg.convert_layout %d : tensor<128x16xf32, #blocked> -> tensor<128x16xf32, #blocked4>
    %p0 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    %p1 = tt.addptr %p0, %a2 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
    %p2 = tt.broadcast %p1 : tensor<128x1x!tt.ptr<f32>, #blocked1> -> tensor<128x16x!tt.ptr<f32>, #blocked1>
    %p3 = ttg.convert_layout %p2 : tensor<128x16x!tt.ptr<f32>, #blocked1> -> tensor<128x16x!tt.ptr<f32>, #blocked4>
    %p4 = tt.addptr %p3, %a10 : tensor<128x16x!tt.ptr<f32>, #blocked4>, tensor<128x16xi32, #blocked4>
    tt.store %p4, %d0 : tensor<128x16x!tt.ptr<f32>, #blocked4>
    tt.return
  }
}
