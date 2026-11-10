// T11: the packed-dot landing sequence (design doc 22.22).
//
// `tl.dot` cannot be lowered with a general contract yet: fmfma's x is one
// column per PE (B[:, q]) and no measured placement primitive produces that
// from a row-major buffer. The narrowing decided in 22.22 is that the caller
// supplies B pre-packed per PE column; this fixture is the exact mncore-level
// sequence such a lowering must emit, and it is what T11 pins.
tt.func @packed_gemm() {
  %a0 = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 1 : i32, flags = 0 : i32}> : () -> !mncore.token
  %a1 = "mncore.lbop"() <{mem = 0 : i32, addr = 64 : i64, width = 1 : i32, flags = 0 : i32}> : () -> !mncore.token
  %a2 = "mncore.lbop"() <{mem = 0 : i32, addr = 128 : i64, width = 1 : i32, flags = 0 : i32}> : () -> !mncore.token
  %a3 = "mncore.lbop"() <{mem = 0 : i32, addr = 192 : i64, width = 1 : i32, flags = 0 : i32}> : () -> !mncore.token
  %alo = "mncore.slab.load"(%a0, %a1, %a2, %a3) <{mode = 3 : i64, zmask = -1 : i64, wmask = -1 : i64}> : (!mncore.token, !mncore.token, !mncore.token, !mncore.token) -> vector<8xf32>
  %b0 = "mncore.lbop"() <{mem = 0 : i32, addr = 256 : i64, width = 1 : i32, flags = 0 : i32}> : () -> !mncore.token
  %b1 = "mncore.lbop"() <{mem = 0 : i32, addr = 320 : i64, width = 1 : i32, flags = 0 : i32}> : () -> !mncore.token
  %b = "mncore.vec.load"(%b0, %b1) <{mode = 3 : i64, zmask = -1 : i64, wmask = -1 : i64}> : (!mncore.token, !mncore.token) -> vector<4xf32>
  %c = "mncore.gemm.tile"(%alo, %alo, %b, %alo) <{face = 0 : i64}> : (vector<8xf32>, vector<8xf32>, vector<4xf32>, vector<8xf32>) -> vector<8xf32>
  tt.return
}
