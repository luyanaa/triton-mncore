//===- TritonGPUToMNCore.cpp - TTGIR to mncore conversion -----------------===//
//
// Prototype TTGIR -> mncore lowering. It is intentionally strict: only
// straight-line kernels whose tensors give each PE one 2-longword slice are
// accepted. Everything else is rejected with a diagnostic rather than lowered
// through an approximate GPU path.
//
//===----------------------------------------------------------------------===//

#include "mncore/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MNCore/IR/MNCoreOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <optional>
#include <utility>

using namespace mlir;

namespace ttg = mlir::triton::gpu;

namespace {

enum class Distribution { Broadcast, MAB, MAB4, Distributed };

struct DistributionInfo {
  Distribution kind;
  VectorType vectorType;
};

// MN-Core executes one expression across all 64 PEs (16 MABs x 4 PEs). The
// distribution mode is a property of *which lane bits are replicated*, not of
// the element order inside a PE (the frontend may use strided, contiguous or
// plane-major orders):
//
//   mode         replicated lane/warp bits     distinct lanes   elements
//   Broadcast    all six                      1                1 * lanes
//   MAB          bits 2..5 (all MAB bits)     4 (p)            4 * lanes
//   MAB4         bits 2..3 (within 4 MABs)    16 (p, group)    16 * lanes
//   Distributed  none                         64               64 * lanes
//
// This is exactly SDM 3.6.8: l1bmm repeats the 4 PE values across all 16 MABs,
// l1bmm4 shares them across groups of 4 MABs, and l1bmd gives every PE a
// distinct value. Anything else is rejected rather than silently replicated.
//
// Rank: the criterion is rank-agnostic. A rank-2 (or higher) tile is accepted
// iff its linear layout has the same replicated-bit structure; the *logical*
// element order of the tensor -- row-major over its shape -- is then the
// physical order the DMA path uses. That is a contract, not an accident: the
// PE register file is a flat sequence of longwords and every L1BM transfer is
// a flat stream, so a 2-D tile is only expressible when flattening it
// row-major reproduces the stream. `checkFullTile` below enforces the other
// half of it (the pointer expression must be exactly base + row-major rank).
static FailureOr<DistributionInfo> analyzeDistribution(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || tensor.getRank() < 1)
    return failure();
  if (!isa<ttg::DistributedEncodingTrait>(tensor.getEncoding()))
    return failure();

  Type element = tensor.getElementType();
  unsigned lanes;
  if (element.isF32())
    lanes = 4;
  else if (element.isF64())
    lanes = 2;
  else if (element.isF16())
    lanes = 8;
  else
    return failure();

  auto linear = ttg::toLinearEncoding(tensor);
  auto ll = linear.getLinearLayout();
  // A basis is one vector per input dimension; a bit is replicated when the
  // whole vector is zero. Comparing full vectors (not just dimension 0) is
  // what makes the test rank-independent.
  auto basesOf = [&](StringRef name) {
    SmallVector<SmallVector<int32_t>> out;
    auto key = StringAttr::get(tensor.getContext(), name);
    auto it = ll.getBases().find(key);
    if (it == ll.getBases().end())
      return out;
    for (const auto &basis : it->second) {
      SmallVector<int32_t> b(basis.begin(), basis.end());
      // Multi-dim bases may be padded/truncated by the encoding; a missing
      // dimension is a zero component.
      b.resize(tensor.getRank(), 0);
      out.push_back(b);
    }
    return out;
  };
  auto reg = basesOf("register");
  auto lane = basesOf("lane");
  auto warp = basesOf("warp");

  // 4 f32 (2 f64 / 8 f16) per PE: exactly log2(lanes) register bits.
  if ((1u << reg.size()) != lanes)
    return failure();

  // 64 PE lanes = 32 lane bits + 1 warp bit.
  SmallVector<SmallVector<int32_t>> bits;
  bits.append(lane.begin(), lane.end());
  bits.append(warp.begin(), warp.end());
  if (bits.size() > 6)
    return failure();
  // Canonicalized replicated layouts may drop the zero lane/warp bits.
  bits.resize(6, SmallVector<int32_t>(tensor.getRank(), 0));
  unsigned nonzero = 0;
  for (const auto &b : bits) {
    bool zero = llvm::all_of(b, [](int32_t v) { return v == 0; });
    if (!zero)
      ++nonzero;
  }

  auto isZero = [&](unsigned i) {
    return llvm::all_of(bits[i], [](int32_t v) { return v == 0; });
  };
  int64_t elements = 1;
  for (int64_t d : tensor.getShape())
    elements *= d;
  VectorType vectorType = VectorType::get({lanes}, element);

  Distribution kind;
  if (nonzero == 0) {
    if (elements != static_cast<int64_t>(lanes))
      return failure();
    kind = Distribution::Broadcast;
  } else if (nonzero == 2 && !isZero(0) && !isZero(1) && isZero(2) &&
             isZero(3) && isZero(4) && isZero(5)) {
    if (elements != 4LL * lanes)
      return failure();
    kind = Distribution::MAB;
  } else if (nonzero == 4 && !isZero(0) && !isZero(1) && isZero(2) &&
             isZero(3) && !isZero(4) && !isZero(5)) {
    if (elements != 16LL * lanes)
      return failure();
    kind = Distribution::MAB4;
  } else if (nonzero == 6) {
    if (elements != 64LL * lanes)
      return failure();
    kind = Distribution::Distributed;
  } else {
    return failure();
  }
  return DistributionInfo{kind, vectorType};
}

static FailureOr<VectorType> convertTensorType(Type type) {
  auto info = analyzeDistribution(type);
  if (failed(info))
    return failure();
  return info->vectorType;
}

//===----------------------------------------------------------------------===//
// Full-tile pointer matching
//===----------------------------------------------------------------------===//
//
// The global-memory path accepts exactly one access shape: a whole contiguous
// tile whose element at logical row-major index i is `base + i`, where `base`
// is a kernel buffer argument. Only compile-time offsets are accepted.
//
// The offset is matched by *evaluating* it, not by pattern-matching a fixed
// tree. TTGIR expands a natural index expression such as
// `a_ptr + rm[:, None] * C + rn[None, :]` into make_range / expand_dims /
// broadcast / convert_layout / addi / muli, and the exact folding differs
// between Triton versions and between 1-D and 2-D kernels. Matching the tree
// would make which kernels compile depend on that shape; evaluating the tensor
// and requiring offset[i] == i accepts every spelling of the same access and
// nothing else.

struct IdxTile {
  SmallVector<int64_t> vals; // row-major over the logical shape
};

static SmallVector<int64_t> shapeOf(Value v) {
  auto rt = dyn_cast<RankedTensorType>(v.getType());
  if (!rt)
    return {};
  return SmallVector<int64_t>(rt.getShape().begin(), rt.getShape().end());
}

static int64_t numel(ArrayRef<int64_t> shape) {
  int64_t n = 1;
  for (int64_t d : shape)
    n *= d;
  return n;
}

static SmallVector<int64_t> unravel(ArrayRef<int64_t> shape, int64_t lin) {
  SmallVector<int64_t> idx(shape.size());
  for (int i = (int)shape.size() - 1; i >= 0; --i) {
    idx[i] = lin % shape[i];
    lin /= shape[i];
  }
  return idx;
}

static int64_t ravel(ArrayRef<int64_t> shape, ArrayRef<int64_t> idx) {
  int64_t lin = 0;
  for (size_t i = 0; i < shape.size(); ++i)
    lin = lin * shape[i] + idx[i];
  return lin;
}

// Map a source tile onto a (possibly broadcast) destination shape.
static std::optional<IdxTile> broadcastVals(const IdxTile &in,
                                            ArrayRef<int64_t> srcShape,
                                            ArrayRef<int64_t> dstShape) {
  if (srcShape.size() != dstShape.size())
    return std::nullopt;
  IdxTile out;
  int64_t n = numel(dstShape);
  out.vals.reserve(n);
  for (int64_t lin = 0; lin < n; ++lin) {
    auto idx = unravel(dstShape, lin);
    for (size_t d = 0; d < idx.size(); ++d)
      if (srcShape[d] == 1)
        idx[d] = 0;
    out.vals.push_back(in.vals[ravel(srcShape, idx)]);
  }
  return out;
}

static std::optional<IdxTile> evalIndex(Value v, unsigned depth = 0) {
  if (depth > 24)
    return std::nullopt;
  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;

  if (auto range = dyn_cast<triton::MakeRangeOp>(op)) {
    IdxTile t;
    for (int64_t i = range.getStart(); i < range.getEnd(); ++i)
      t.vals.push_back(i);
    return t;
  }
  if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
    auto attr = dyn_cast<DenseElementsAttr>(cst.getValue());
    if (!attr || !attr.getElementType().isInteger(32))
      return std::nullopt;
    IdxTile t;
    for (auto el : attr.getValues<APInt>())
      t.vals.push_back(el.getSExtValue());
    return t;
  }
  if (auto sp = dyn_cast<triton::SplatOp>(op)) {
    auto ic = sp.getSrc().getDefiningOp<arith::ConstantOp>();
    if (!ic)
      return std::nullopt;
    auto ia = dyn_cast<IntegerAttr>(ic.getValue());
    if (!ia)
      return std::nullopt;
    IdxTile t;
    t.vals.assign(numel(shapeOf(sp.getResult())), ia.getInt());
    return t;
  }
  if (auto ed = dyn_cast<triton::ExpandDimsOp>(op)) {
    // Inserting a size-1 dimension leaves the row-major value order alone.
    return evalIndex(ed.getSrc(), depth + 1);
  }
  if (auto bc = dyn_cast<triton::BroadcastOp>(op)) {
    auto in = evalIndex(bc.getSrc(), depth + 1);
    if (!in)
      return std::nullopt;
    return broadcastVals(*in, shapeOf(bc.getSrc()), shapeOf(bc.getResult()));
  }
  if (auto cl = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    // A relayout permutes values between threads but not within the logical
    // tensor, so an index tile is unchanged by it.
    auto in = evalIndex(cl.getSrc(), depth + 1);
    if (!in)
      return std::nullopt;
    if (numel(shapeOf(cl.getSrc())) != numel(shapeOf(cl.getResult())))
      return std::nullopt;
    return in;
  }
  if (isa<arith::AddIOp, arith::MulIOp>(op)) {
    auto a = evalIndex(op->getOperand(0), depth + 1);
    auto b = evalIndex(op->getOperand(1), depth + 1);
    if (!a || !b || a->vals.size() != b->vals.size())
      return std::nullopt;
    IdxTile t;
    for (size_t i = 0; i < a->vals.size(); ++i)
      t.vals.push_back(isa<arith::AddIOp>(op) ? a->vals[i] + b->vals[i]
                                              : a->vals[i] * b->vals[i]);
    return t;
  }
  return std::nullopt;
}

struct PtrTile {
  Value base;
  SmallVector<int64_t> off;
};

static std::optional<PtrTile> evalPtrTile(Value v, unsigned depth = 0) {
  if (depth > 24)
    return std::nullopt;
  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;

  if (auto sp = dyn_cast<triton::SplatOp>(op)) {
    auto rt = dyn_cast<RankedTensorType>(sp.getResult().getType());
    if (!rt || !isa<triton::PointerType>(rt.getElementType()))
      return std::nullopt;
    PtrTile t;
    t.base = sp.getSrc();
    t.off.assign(numel(rt.getShape()), 0);
    return t;
  }
  if (auto ap = dyn_cast<triton::AddPtrOp>(op)) {
    auto in = evalPtrTile(ap.getPtr(), depth + 1);
    if (!in)
      return std::nullopt;
    auto idx = evalIndex(ap.getOffset());
    if (!idx || idx->vals.size() != in->off.size())
      return std::nullopt;
    for (size_t i = 0; i < in->off.size(); ++i)
      in->off[i] += idx->vals[i];
    return in;
  }
  if (auto cl = dyn_cast<ttg::ConvertLayoutOp>(op))
    return evalPtrTile(cl.getSrc(), depth + 1);
  if (auto bc = dyn_cast<triton::BroadcastOp>(op)) {
    auto in = evalPtrTile(bc.getSrc(), depth + 1);
    if (!in)
      return std::nullopt;
    IdxTile ids;
    ids.vals = in->off;
    auto out = broadcastVals(ids, shapeOf(bc.getSrc()), shapeOf(bc.getResult()));
    if (!out)
      return std::nullopt;
    PtrTile t;
    t.base = in->base;
    t.off = std::move(out->vals);
    return t;
  }
  return std::nullopt;
}

// Match a whole contiguous tile: `base_ptr + <row-major 0..N-1>`.
static bool matchFullTile(Value ptr, Value &baseArg, int64_t &N) {
  auto t = evalPtrTile(ptr);
  if (!t || !isa<BlockArgument>(t->base))
    return false;
  N = t->off.size();
  if (N <= 0)
    return false;
  for (int64_t i = 0; i < N; ++i)
    if (t->off[i] != i)
      return false;
  baseArg = t->base;
  return true;
}

// Ops that only compute an address or an index: their results are integers or
// pointers, never PE data, and `matchFullTile` re-evaluates them from the
// load/store that consumes them.
static bool isIndexOnly(Operation *op) {
  if (op->getNumResults() != 1)
    return false;
  auto rt = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!rt)
    return false;
  Type el = rt.getElementType();
  return el.isIntOrIndex() || isa<triton::PointerType>(el);
}

//===----------------------------------------------------------------------===//
// tt.dot
//===----------------------------------------------------------------------===//
//
// The MAU has no matrix-matrix product. SDM 3.6.9.1 (p.111) lists exactly two
// matrix forms, mfma and mmul (mfma with y forced to 0), and each takes one
// matrix face plus PE-memory vectors. For f32, 3.6.9.4 (p.114) defines one
// fmfma as, per MAB,
//
//   dst[i] = sum(j=0..3) A[i][2j] * x[j] + y[i],   i = 0..7
//
// where A is the 8x8 logical single-precision face (only the even logical
// columns 0,2,4,6 are used, so the effective matrix is 8x4), x is 4 f32 at
// *word* width (one element per PE of the MAB), and y/dst are 8 f32 at
// *longword* width (two per PE). The instruction drives 4 cycles with the same
// A and a cycle-dependent x/y/dst address, so four B columns can be consumed
// per instruction -- the 8x4-by-4x4 block is emergent from the per-cycle
// pseudocode, not a documented matrix-matrix operation.
//
// A kernel-level GEMM is therefore mwrite(A) then one fmfma per 4 columns of
// B per 8-row M block, chaining K in blocks of 4 through y.
//
// A TTGIR `tl.dot` cannot feed that sequence directly: with a blocked parent,
// `fmaDotToLinearLayout` puts the whole K dimension in every thread's registers
// (32 f32 per thread for a 32x32x32 dot), while the MAU consumes K one element
// per PE, and no measured primitive can produce that per-PE assignment. The
// blocked GEMM is therefore expressed as a KB x NB grid of `mncore.gemm.tile`
// ops over *pre-arranged* buffers (design doc 22.30, 22.32, 22.34), and a dot
// whose operands are not such buffers is rejected with a diagnostic that cites
// the clause it violates.
//
// The dot carries the *logical* 2-D shapes -- A (M,K), B (K,N), C (M,N) -- so
// the type system checks the algebra and the pass only adds the hardware
// constraints. M is fixed at 128: one gemm.tile grid is exactly the 16 MABs
// times the 8 face rows, and K and N must be multiples of the 4 the MAU
// consumes per block. KB = K/4 and NB = N/4 are then the block counts.
//
// The buffers *at the operand pointers* are the pre-arranged, block-major
// arrays (design doc 22.30, 22.32, 22.34) -- not the logical tiles:
//
//   A'  1024 * KB f32   block kb = 8 planes of 64 longwords (2x the logical
//                       128x4 block: the face uses only even columns, so each
//                       value sits alone in a longword's low half);
//                       plane r, longword 4m+q = A[8m+r][4kb+q], low half
//   B'  256 * KB*NB f32 block (kb,nb) = 2 planes of 64 longwords (16x the
//                       logical 4x4 block: l1bmd gives each of the 64 PEs its
//                       own 4 f32, so one MAB's 4x4 x operand is repeated 16
//                       times), deployment-time layout, x per PE
//   C   512 * NB f32    block nb = 4 planes, i.e. 8 f32 per PE -- exactly the
//                       logical 128x4 C block, so no padding
//
// The compiler therefore reads 2x numel(A), 16x numel(B) and numel(C) f32 from
// the three base pointers. A reader that treats those pointers as plain tiles
// will over-read; the sizes are part of the kernel-argument contract, like the
// whole-tile rule, and A is verified at dump level by
// scripts/check_a_layout.py (22.33.4).
static FailureOr<std::pair<unsigned, unsigned>>
slicedDotShape(triton::DotOp dot) {
  auto shape2 = [](Value v) -> std::optional<std::pair<int64_t, int64_t>> {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || t.getRank() != 2 || !t.getElementType().isF32())
      return std::nullopt;
    return std::make_pair(t.getShape()[0], t.getShape()[1]);
  };
  auto a = shape2(dot.getA()), b = shape2(dot.getB());
  auto c = shape2(dot.getC()), r = shape2(dot.getResult());
  if (!a || !b || !c || !r)
    return failure();
  int64_t M = a->first, K = a->second, N = c->second;
  if (M != 128 || c->first != M || b->first != K || b->second != N ||
      r->first != M || r->second != N)
    return failure();
  if (K % 4 != 0 || N % 4 != 0)
    return failure();
  return std::make_pair((unsigned)(K / 4), (unsigned)(N / 4));
}

static LogicalResult rejectDot(triton::DotOp dot) {
  auto aTy = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(dot.getB().getType());
  auto cTy = dyn_cast<RankedTensorType>(dot.getC().getType());
  if (!aTy || !bTy || !cTy || aTy.getRank() != 2 || bTy.getRank() != 2 ||
      cTy.getRank() != 2)
    return dot.emitError()
           << "tt.dot on MN-Core requires rank-2 A/B/C tensors";

  // Precision: the fmfma contract above is f32. The other MAU precisions have
  // different shapes and are not wired (h = 16x16 f16 with single-precision
  // accumulation, 3.6.9.6; d = 4x4 f64 with the u|d row half-select, 3.6.9.2).
  Type element = cTy.getElementType();
  if (!element.isF32() || !aTy.getElementType().isF32() ||
      !bTy.getElementType().isF32())
    return dot.emitError()
           << "tt.dot on MN-Core is f32 only: the implemented MAU form is "
              "fmfma (SDM 3.6.9.4), an 8x4-by-4 block with K=4. "
           << element
           << " would need the h (hmfma, 16x16 f16 with single-precision "
              "accumulate, 3.6.9.6) or d (dmfma with the u|d row half-select, "
              "4x4 f64, 3.6.9.2) form, neither of which is wired";
  if (dot.getInputPrecision() != triton::InputPrecision::IEEE ||
      dot.getMaxNumImpreciseAcc() != 0)
    return dot.emitError()
           << "tt.dot on MN-Core is ieee only: the MAU has no reduced-"
              "precision input mode, and TF32/FP8 operand modes do not exist";

  // Tiling: the matrix face covers 8 rows (M) and fmfma covers 4 B columns (N)
  // and K=4 per pass, all per MAB.
  int64_t M = cTy.getShape()[0], N = cTy.getShape()[1], K = aTy.getShape()[1];
  if (M % 8 != 0 || N % 4 != 0 || K % 4 != 0)
    return dot.emitError()
           << "tt.dot " << M << "x" << K << "x" << N
           << " does not tile the MAU: the matrix face is 8 rows (M) and one "
              "fmfma covers 4 columns of B (N) and 4 of K, so M, N and K must "
              "be multiples of 8, 4 and 4";

  auto aLayout = aTy.getEncoding();
  auto bLayout = bTy.getEncoding();
  return dot.emitError()
         << "tt.dot " << M << "x" << K << "x" << N
         << " is not lowerable by the blocked-GEMM path: the sequence is one "
            "mwrite of an 8x4 A tile into a matrix face and one fmfma per 4 "
            "columns of B (x at word width, 4 f32 per PE; y/dst at longword "
            "width, 8 f32 per PE), chaining K in blocks of 4 through y "
            "(mncore_intrinsics_design.md section 22). One gemm.tile grid "
            "covers exactly M = 128 (16 MABs x 8 face rows) with K and N "
            "multiples of 4, and its operands must point at the pre-arranged "
            "block buffers (2x numel for A, 16x for B). This dot carries A: "
         << aLayout << ", B: " << bLayout
         << "; a blocked-parent dot-operand layout also replicates the whole K "
            "dimension into every thread's registers, whereas the MAU "
            "consumes K one element per PE of a MAB, so even a shape-"
            "compatible dot must be given the mncore operand layout first";
}

static LogicalResult convertFunction(triton::FuncOp fn) {
  if (fn.isExternal())
    return success();
  MLIRContext *ctx = fn.getContext();
  Location loc = fn.getLoc();

  if (!fn.getBody().hasOneBlock())
    return fn.emitError()
           << "tritongpu-to-mncore requires a single-block, straight-line "
              "kernel (the MN-Core backend rejects branches)";
  if (!fn.getFunctionType().getResults().empty())
    return fn.emitError() << "kernel must return void";

  // A sliced dot owns its operand loads, the relayouts that put them into the
  // dot-operand encoding, and its result store. The dot branch below emits the
  // DMAs, the per-block slab loads, the chained gemm.tile ops and the
  // per-block writeback itself, so the ordinary load/store/convert_layout
  // paths must not run for those operations -- and any *other* use of those
  // values would need that ordinary path, which has no meaning for a
  // pre-arranged buffer. Resolving the ownership here keeps the diagnostic on
  // the right op and keeps the walk free of half-materialised tiles.
  //
  // DotPlan holds the two block counts, the three operand loads and the store
  // that writes C back.
  struct DotPlan {
    unsigned kb, nb;
    triton::LoadOp a, b, c;
    triton::StoreOp out;
  };
  DenseMap<Operation *, DotPlan> dotPlans;
  DenseSet<Value> dotLoads;
  DenseSet<Operation *> dotStores;
  DenseSet<Operation *> dotRelayouts;
  for (Operation &op : fn.getBody().front()) {
    auto dot = dyn_cast<triton::DotOp>(op);
    if (!dot)
      continue;
    auto blocks = slicedDotShape(dot);
    // A synthesised zero accumulator is a property of the operand, not of the
    // shape; name it first so `tl.dot(a, b)` gets an actionable message rather
    // than "C must come from a whole-tile tt.load".
    if (dot.getC().getDefiningOp<arith::ConstantOp>())
      return dot.emitError()
             << "a sliced dot cannot start from a compiler-synthesised zero "
                "accumulator: the grid reads each C block from its buffer as "
                "the initial y, and there is no immediate-to-PE-data "
                "primitive to build a zero slab (the MAU's zeroing form is "
                "mmul, matvec with y forced to 0, which this path does not "
                "emit). Pass an explicit accumulator tile, i.e. load a "
                "zeroed buffer";
    if (failed(blocks))
      return rejectDot(dot);
    auto loadOf = [&](Value v, StringRef name,
                      triton::LoadOp &out) -> LogicalResult {
      // The dot operand encoding is a relayout away from the loaded tile in
      // every real TTGIR; it only permutes registers, and the value is never
      // materialised (the grid reads the buffer, not the tile), so one
      // convert_layout is carried along and dropped with the load.
      if (auto cl = v.getDefiningOp<ttg::ConvertLayoutOp>()) {
        if (!cl.getResult().hasOneUse())
          return dot.emitError()
                 << "a sliced dot's " << name
                 << " relayout must have the dot as its only user: the value "
                    "is not materialised";
        dotRelayouts.insert(cl);
        v = cl.getSrc();
      }
      auto ld = v.getDefiningOp<triton::LoadOp>();
      if (!ld || !ld.getResult().hasOneUse())
        return dot.emitError()
               << "a sliced dot's " << name
               << " must come from a whole-tile tt.load whose only user is "
                  "(a relayout feeding) the dot: the block grid is defined "
                  "over pre-arranged buffers (design doc 22.30), so the "
                  "compiler places those buffers and cannot consume a "
                  "computed operand";
      out = ld;
      return success();
    };
    DotPlan plan{blocks->first, blocks->second, nullptr, nullptr, nullptr,
                 nullptr};
    if (failed(loadOf(dot.getA(), "A", plan.a)) ||
        failed(loadOf(dot.getB(), "B", plan.b)) ||
        failed(loadOf(dot.getC(), "C", plan.c)))
      return failure();
    // The result is stored through a relayout in every real TTGIR (the dot
    // produces the C encoding, the store wants the tile encoding); like the
    // operand relayouts it is dead once the grid is emitted.
    Value stored = dot.getResult();
    if (stored.hasOneUse())
      if (auto cl = dyn_cast<ttg::ConvertLayoutOp>(*stored.getUsers().begin())) {
        dotRelayouts.insert(cl);
        stored = cl.getResult();
      }
    if (!stored.hasOneUse())
      return dot.emitError()
             << "a sliced dot's result must have exactly one use, the tt.store "
                "that writes the C buffer: the block grid writes C block by "
                "block through L1BM, so there is no register-level tile value "
                "to hand to another operation";
    plan.out = dyn_cast<triton::StoreOp>(*stored.getUsers().begin());
    if (!plan.out || plan.out.getValue() != stored)
      return dot.emitError()
             << "a sliced dot's result must be consumed by a tt.store directly";
    if (plan.out.getMask())
      return plan.out.emitError()
             << "a masked store cannot write a sliced dot's C buffer: the grid "
                "writes whole 512xf32 blocks with slab transfers";
    dotLoads.insert(plan.a.getResult());
    dotLoads.insert(plan.b.getResult());
    dotLoads.insert(plan.c.getResult());
    dotStores.insert(plan.out);
    dotPlans[&op] = plan;
  }

  // Masking is likewise a property of the access, not of one PE: the global
  // path moves the whole 256xf32 tile with one DMA descriptor (src_arg names
  // the buffer, the two plane records carry the byte offsets) and reads it
  // into each PE's slot with one distributed l1bmd. There is no per-element
  // predicate anywhere in that path -- `vec.*` wmask/zmask select *operand
  // lanes*, four per PE, and cannot mask an individual tile element without
  // also masking three others. Checking here rather than in the walk keeps
  // the diagnostic on the `tt.load`/`tt.store` even when the mask (or its
  // `other` constant) was materialised earlier in the block.
  for (Operation &op : fn.getBody().front()) {
    if (auto load = dyn_cast<triton::LoadOp>(op))
      if (load.getMask() || load.getOther())
        return load.emitError()
               << "masked tile loads are not lowerable: the 256xf32 global "
                  "path moves the whole tile with one DMA descriptor and "
                  "reads it with one distributed l1bmd, and `vec.*` masks "
                  "select operand lanes (4 f32 per PE), not tile elements. A "
                  "partial tile has to be expressed as a smaller whole tile "
                  "or a scalar path";
    if (auto store = dyn_cast<triton::StoreOp>(op))
      if (store.getMask())
        return store.emitError()
               << "masked tile stores are not lowerable: the 256xf32 global "
                  "path writes the whole tile back with two l2bm point "
                  "transfers, and `vec.*` masks select operand lanes (4 f32 "
                  "per PE), not tile elements";
  }

  SmallVector<Type> args;
  // Buffer arguments are not resolved to DRAM addresses here: the .vsm
  // address field is a static immediate, so each use becomes a placeholder 0
  // plus a relocation record naming the argument (mncore_kernel_abi_v1.md
  // §13). The loader binds the argument to a device DRAM base at run time.
  DenseMap<Value, unsigned> ptrArgs;
  uint64_t argIndex = 0;
  for (Type t : fn.getFunctionType().getInputs()) {
    if (auto vec = convertTensorType(t); succeeded(vec)) {
      args.push_back(*vec);
    } else if (auto ptr = dyn_cast<triton::PointerType>(t);
               ptr && ptr.getPointeeType().isF32()) {
      args.push_back(IntegerType::get(ctx, 64));
      ptrArgs[fn.getBody().front().getArgument(argIndex)] =
          (unsigned)argIndex;
    } else {
      return fn.emitError()
             << "unsupported kernel argument type " << t
             << "; expected a blocked/linear tensor with 4xf32 / 2xf64 / "
                "8xf16 per PE (any rank, flattened row-major), or an *fp32 "
                "buffer";
    }
    ++argIndex;
  }

  auto newType = FunctionType::get(ctx, args, {});
  OpBuilder b(fn);
  auto newFn = func::FuncOp::create(b, loc, fn.getSymName(), newType);
  newFn.setVisibility(fn.getVisibility());

  Block &oldBody = fn.getBody().front();
  Block *newBody = newFn.addEntryBlock();
  DenseMap<Value, Value> map;
  struct AllocInfo {
    uint64_t base;
    uint64_t longwords;
  };
  DenseMap<Value, AllocInfo> allocs;
  uint64_t nextL2 = 0;
  uint32_t nextTag = 1;
  for (unsigned i = 0; i < oldBody.getNumArguments(); ++i)
    map[oldBody.getArgument(i)] = newBody->getArgument(i);

  OpBuilder body = OpBuilder::atBlockEnd(newBody);
  uint64_t nextL1 = 0;
  // True once a tile has been written back. The writeback (`l2bm@`) and a
  // following tile's DMA load both drive the L2BM local port, and assemble3's
  // hazard checker rejects the load when the writeback is two steps old:
  // "A L2BM_Local memory port is unavailable" on the first `l2bmb` after it.
  // Measured on a two-tile kernel: two nop steps of separation assemble, one
  // does not. Emitting the two fences here (rather than at the end of the
  // store path) keeps them in front of the DMA no matter what the next tile
  // starts with.
  bool wroteBack = false;

  for (Operation &op : oldBody) {
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (alloc.getSrc())
        return alloc.emitError()
               << "local_alloc with an initializer is not supported yet";
      auto memdesc = dyn_cast<ttg::MemDescType>(alloc.getType());
      if (!memdesc)
        return alloc.emitError() << "local_alloc result is not a memdesc";
      uint64_t elements = 1;
      for (int64_t d : memdesc.getShape())
        elements *= d;
      uint64_t bits = memdesc.getElementType().getIntOrFloatBitWidth();
      uint64_t longwords = std::max<uint64_t>(2, (elements * bits + 63) / 64);
      // Align every allocation to the strictest mode (l1bmd/MAB4).
      nextL1 = (nextL1 + 63) & ~uint64_t{63};
      allocs[alloc.getResult()] = {nextL1, longwords};
      nextL1 += longwords;
      continue;
    }

    if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
      auto it = allocs.find(load.getSrc());
      if (it == allocs.end())
        return load.emitError()
               << "local_load source is not an allocated L1BM buffer";
      auto dist = analyzeDistribution(load.getType());
      if (failed(dist))
        return load.emitError()
               << "local_load layout is neither a full 64-PE distribution "
                  "(64 distinct lane ids, elements == 64*lanes with lanes the "
                  "per-PE width: 4 f32 / 2 f64 / 8 f16) nor a broadcast (one "
                  "thread, elements == lanes). Rank does not matter, only the "
                  "replicated lane bits and the total element count: "
               << load.getType();
      auto tokenType = mncore::TokenType::get(ctx);
      Location loc = load.getLoc();
      Value value;
      auto modeOf = [](Distribution kind) {
        switch (kind) {
        case Distribution::Broadcast:
          return mncore::L1Mode::Broadcast;
        case Distribution::MAB:
          return mncore::L1Mode::MAB;
        case Distribution::MAB4:
          return mncore::L1Mode::MAB4;
        case Distribution::Distributed:
          return mncore::L1Mode::Distributed;
        }
        llvm_unreachable("unknown distribution");
      };
      if (dist->kind != Distribution::Distributed) {
        // Broadcast/MAB/MAB4 have a double-longword form: one transfer,
        // width=2. The MAB/MAB4 replication is encoded in the linear layout.
        auto lb = mncore::LbOpOp::create(body, loc, tokenType, 0u,
                                         it->second.base, 2u, 0u);
        auto none = mncore::NoneOp::create(body, loc, tokenType);
        value = mncore::VecLoadOp::create(body, loc, dist->vectorType,
                                          modeOf(dist->kind), lb, none);
      } else {
        // l1bmd has no double-longword form: one transfer per longword plane.
        // The shared layout contract places the high plane immediately after
        // the 64-longword low plane.
        uint64_t plane = it->second.longwords / 2;
        auto lo = mncore::LbOpOp::create(body, loc, tokenType, 0u,
                                         it->second.base, 1u, 0u);
        auto hi = mncore::LbOpOp::create(body, loc, tokenType, 0u,
                                         it->second.base + plane, 1u, 0u);
        value = mncore::VecLoadOp::create(
            body, loc, dist->vectorType, mncore::L1Mode::Distributed, lo, hi);
      }
      map[load.getResult()] = value;
      continue;
    }

    if (auto add = dyn_cast<arith::AddFOp>(op)) {
      auto a = map.find(add.getLhs());
      auto c = map.find(add.getRhs());
      auto vec = convertTensorType(add.getType());
      if (a == map.end() || c == map.end() || failed(vec))
        return add.emitError()
               << "addf operands are not mncore vectors: elementwise "
                  "arithmetic runs on one 4xf32 vector per PE (a 256xf32 tile "
                  "across the 64 PEs), and both operands must be whole-tile "
                  "loads of that width -- the wider 512xf32 slab is a matrix "
                  "operand only and feeds gemm.tile, not arith";
      auto res = mncore::VecAddOp::create(body, add.getLoc(), *vec, a->second,
                                               c->second);
      map[add.getResult()] = res;
      continue;
    }
    if (auto mul = dyn_cast<arith::MulFOp>(op)) {
      auto a = map.find(mul.getLhs());
      auto c = map.find(mul.getRhs());
      auto vec = convertTensorType(mul.getType());
      if (a == map.end() || c == map.end() || failed(vec))
        return mul.emitError() << "mulf operands are not mncore vectors";
      auto res = mncore::VecMulOp::create(body, mul.getLoc(), *vec, a->second,
                                               c->second);
      map[mul.getResult()] = res;
      continue;
    }
    if (auto maxOp = dyn_cast<arith::MaximumFOp>(op)) {
      auto a = map.find(maxOp.getLhs());
      auto c = map.find(maxOp.getRhs());
      auto vec = convertTensorType(maxOp.getType());
      if (a == map.end() || c == map.end() || failed(vec))
        return maxOp.emitError() << "maximumf operands are not mncore vectors";
      auto res = mncore::VecMaxOp::create(body, maxOp.getLoc(), *vec, a->second,
                                               c->second);
      map[maxOp.getResult()] = res;
      continue;
    }
    if (auto minOp = dyn_cast<arith::MinimumFOp>(op)) {
      auto a = map.find(minOp.getLhs());
      auto c = map.find(minOp.getRhs());
      auto vec = convertTensorType(minOp.getType());
      if (a == map.end() || c == map.end() || failed(vec))
        return minOp.emitError() << "minimumf operands are not mncore vectors";
      auto res = mncore::VecMinOp::create(body, minOp.getLoc(), *vec, a->second,
                                               c->second);
      map[minOp.getResult()] = res;
      continue;
    }
    if (auto ret = dyn_cast<triton::ReturnOp>(op)) {
      func::ReturnOp::create(body, ret.getLoc());
      continue;
    }

    // Address and index arithmetic: `matchFullTile` re-evaluates the whole
    // pointer expression from the load/store that consumes it, so nothing with
    // an integer or pointer tensor type produces PE data. The backend has no
    // integer data path at all, so dropping these is total; an address that is
    // not compile-time (a runtime value in the chain) makes `matchFullTile`
    // fail at the load/store with a precise diagnostic.
    if (isIndexOnly(&op))
      continue;

    // A relayout of *data* is a no-op in this model only when both sides have
    // the same distribution: the pass never materialises a per-thread
    // placement, it maps every tensor to the PE-wide value its distribution
    // implies. Converting between two different distributions would need the
    // values moved between PEs, which the PE file has no instruction for.
    if (auto cl = dyn_cast<ttg::ConvertLayoutOp>(op)) {
      if (dotRelayouts.contains(cl.getOperation()))
        continue;   // dead once the dot grid is emitted
      auto it = map.find(cl.getSrc());
      if (it == map.end())
        return cl.emitError()
               << "convert_layout source is not an mncore vector";
      auto src = analyzeDistribution(cl.getSrc().getType());
      auto dst = analyzeDistribution(cl.getResult().getType());
      if (failed(src) || failed(dst) || src->kind != dst->kind)
        return cl.emitError()
               << "convert_layout between different distributions is not "
                  "expressible: the PE register file has no shuffle, so a "
                  "relayout has to go through L1BM (local_alloc, local_store, "
                  "local_load). Source: "
               << cl.getSrc().getType() << ", result: " << cl.getResult().getType();
      map[cl.getResult()] = it->second;
      continue;
    }

    if (isa<triton::BroadcastOp, triton::ExpandDimsOp>(op))
      return op.emitError()
             << "replicating a value across PEs is not lowerable: the "
                "distribution modes (l1bmp/l1bmm/l1bmm4) replicate during the "
                "L1BM read, not in registers. Load the tile with the "
                "replicated layout instead of broadcasting a loaded value";

    if (auto dot = dyn_cast<triton::DotOp>(op)) {
      DotPlan &plan = dotPlans.find(&op)->second;
      // The three buffers come from the operand loads' pointer chains; their
      // element counts were fixed by slicedDotShape, so only provenance and
      // the store/C identity remain to check here.
      auto bufferOf = [&](Value ptr, Operation *at, StringRef name,
                          unsigned &arg) -> LogicalResult {
        Value base;
        int64_t n = 0;
        if (!matchFullTile(ptr, base, n))
          return at->emitError()
                 << "a sliced dot's " << name
                 << " must be a whole tile at `base_ptr` whose offsets are "
                    "exactly the row-major index 0..N-1 (1-D "
                    "`base_ptr + tl.arange(0, N)`, or a 2-D "
                    "`base_ptr + rm[:, None] * C + rn[None, :]`): the block "
                    "grid places whole buffers plane by plane, so partial "
                    "addressing cannot be expressed";
        auto itb = ptrArgs.find(base);
        if (itb == ptrArgs.end())
          return at->emitError()
                 << "the " << name << " base is not a kernel buffer argument";
        arg = itb->second;
        return success();
      };
      unsigned aArg = 0, bArg = 0, cArg = 0, oArg = 0;
      if (failed(bufferOf(plan.a.getPtr(), plan.a, "A", aArg)) ||
          failed(bufferOf(plan.b.getPtr(), plan.b, "B", bArg)) ||
          failed(bufferOf(plan.c.getPtr(), plan.c, "C", cArg)) ||
          failed(bufferOf(plan.out.getPtr(), plan.out, "C store", oArg)))
        return failure();
      // The accumulator is read from the C operand's buffer and each block is
      // written to the *store*'s buffer: these may be different arguments,
      // exactly as `tl.dot(a, b, acc)` followed by `tl.store(c_ptr, d)`.

      // One gemm.tile block: A' = 8 planes (a_lo rows 0-3, a_hi rows 4-7),
      // B' = 2 planes (4 f32 per PE, the deployment-time x layout), C = 4
      // planes (8 f32 per PE). The writeback gets a second 4-plane region so
      // a block's DmaStore never races the next block's slab store.
      uint64_t aWords = (uint64_t)plan.kb * 8 * 64;
      uint64_t bWords = (uint64_t)plan.kb * plan.nb * 2 * 64;
      uint64_t cWords = (uint64_t)plan.nb * 4 * 64;
      auto carve = [](uint64_t &next, uint64_t longwords) {
        uint64_t at = llvm::alignTo(next, 64);
        next = at + longwords;
        return at;
      };
      uint64_t l1A = carve(nextL1, aWords);
      uint64_t l1B = carve(nextL1, bWords);
      uint64_t l1C = carve(nextL1, cWords);
      uint64_t l1O = carve(nextL1, cWords);
      uint64_t l2A = carve(nextL2, aWords);
      uint64_t l2B = carve(nextL2, bWords);
      uint64_t l2C = carve(nextL2, cWords);
      uint64_t l2O = carve(nextL2, cWords);
      Location dloc = dot.getLoc();
      if (wroteBack) {
        mncore::FenceOp::create(body, dloc);
        mncore::FenceOp::create(body, dloc);
      }
      // Every buffer is plane-ordered, so placement is a plain 64-longword
      // copy per plane (design doc 22.33: MV moves 64 longwords at a time).
      auto copyPlanes = [&](uint64_t planes, uint64_t l1, uint64_t l2,
                            unsigned arg) {
        for (uint64_t pl = 0; pl < planes; ++pl)
          mncore::DmaOp::create(body, dloc, /*src_space=*/1u,
                                /*src_addr=*/pl * 64, l2 + pl * 64,
                                l1 + pl * 64, 64, nextTag++, 0u, arg);
      };
      copyPlanes(aWords / 64, l1A, l2A, aArg);
      copyPlanes(bWords / 64, l1B, l2B, bArg);
      copyPlanes(cWords / 64, l1C, l2C, cArg);
      // The MV/L2B transfers are asynchronous; two empty steps separate them
      // from the first L1BM read (measured on assemble3's hazard checker).
      mncore::FenceOp::create(body, dloc);
      mncore::FenceOp::create(body, dloc);

      auto tokenType = mncore::TokenType::get(ctx);
      auto f32v8 = VectorType::get({8}, Float32Type::get(ctx));
      auto f32v4 = VectorType::get({4}, Float32Type::get(ctx));
      auto lbops = [&](uint64_t addr, unsigned count) {
        SmallVector<Value> toks;
        for (unsigned j = 0; j < count; ++j)
          toks.push_back(mncore::LbOpOp::create(body, dloc, tokenType, 0u,
                                                addr + j * 64, 1u, 0u));
        return toks;
      };
      for (unsigned n = 0; n < plan.nb; ++n) {
        // C block n is the initial accumulator y of the K chain.
        auto c0 = lbops(l1C + (uint64_t)n * 256, 4);
        Value acc = mncore::SlabLoadOp::create(
            body, dloc, f32v8, mncore::L1Mode::Distributed, c0[0], c0[1],
            c0[2], c0[3]);
        for (unsigned k = 0; k < plan.kb; ++k) {
          auto lo = lbops(l1A + (uint64_t)k * 512, 4);
          auto hi = lbops(l1A + (uint64_t)k * 512 + 256, 4);
          auto bv = lbops(l1B + ((uint64_t)k * plan.nb + n) * 128, 2);
          Value aLo = mncore::SlabLoadOp::create(
              body, dloc, f32v8, mncore::L1Mode::Distributed, lo[0], lo[1],
              lo[2], lo[3]);
          Value aHi = mncore::SlabLoadOp::create(
              body, dloc, f32v8, mncore::L1Mode::Distributed, hi[0], hi[1],
              hi[2], hi[3]);
          // B is the matrix x operand: 4 f32 per PE, read at word width over
          // four cycles, so its two planes must land in the consecutive slot
          // longwords 0 and 1 (mncore.x.load), not in the vector form's `ll`
          // pair (design doc 22.35).
          Value bOp = mncore::XLoadOp::create(
              body, dloc, f32v4, mncore::L1Mode::Distributed, bv[0], bv[1]);
          // K chaining is the accumulator, and the face alternates so a
          // block's matwrite never lands on the face the previous block's
          // matvec is still reading (design doc 22.28).
          acc = mncore::GemmTileOp::create(body, dloc, f32v8,
                                           /*face=*/k % 2, aLo, aHi, bOp, acc);
        }
        auto o = lbops(l1O + (uint64_t)n * 256, 4);
        mncore::SlabStoreOp::create(body, dloc, acc, mncore::L1Mode::Distributed,
                                    o[0], o[1], o[2], o[3], /*set=*/0);
        // Separate the PE collect from the L2B collect that reads L1BM.
        mncore::FenceOp::create(body, dloc);
        for (unsigned pl = 0; pl < 4; ++pl)
          mncore::DmaStoreOp::create(body, dloc,
                                     l1O + (uint64_t)n * 256 + pl * 64,
                                     l2O + (uint64_t)n * 256 + pl * 64,
                                     /*dr_addr=*/(uint64_t)n * 256 + pl * 64,
                                     64, nextTag++, /*l1b=*/0, /*dr_arg=*/oArg);
        wroteBack = true;
        if (n + 1 < plan.nb) {
          // Drain the L2BM local port before the next block's L1BM traffic.
          mncore::FenceOp::create(body, dloc);
          mncore::FenceOp::create(body, dloc);
        }
      }
      continue;
    }

    if (auto load = dyn_cast<triton::LoadOp>(op)) {
      if (dotLoads.contains(load.getResult())) {
        // The dot branch places this buffer itself, block by block.
        if (load.getMask() || load.getOther())
          return load.emitError()
                 << "a sliced dot's operand loads cannot be masked: the block "
                    "grid moves whole pre-arranged buffers, and `vec.*` masks "
                    "select operand lanes, not buffer elements";
        continue;
      }
      if (load.getMask() || load.getOther())
        return load.emitError()
               << "only unmasked full-tile loads are supported";
      Value baseArg;
      int64_t N = 0;
      if (!matchFullTile(load.getPtr(), baseArg, N))
        return load.emitError()
               << "tt.load must be a whole tile at `base_ptr` whose offsets "
                  "are exactly the row-major index 0..N-1 (1-D "
                  "`base_ptr + tl.arange(0, N)`, or a 2-D "
                  "`base_ptr + rm[:, None] * C + rn[None, :]`); dynamic or "
                  "partial addressing is rejected";
      auto itb = ptrArgs.find(baseArg);
      if (itb == ptrArgs.end())
        return load.emitError() << "load base is not a kernel buffer argument";
      auto rt = dyn_cast<RankedTensorType>(load.getType());
      // Two tile widths are supported: 256xf32 (4 f32 = 2 longwords per PE,
      // a `vec`-style value) and 512xf32 (8 f32 = 4 longwords per PE, a
      // slab). The slab form exists because every matrix role of
      // `mncore.gemm.tile` is 4 longwords wide.
      unsigned perPe = rt ? (unsigned)(N / 64) : 0;
      if (!rt || !rt.getElementType().isF32() ||
          (perPe != 4 && perPe != 8))
        return load.emitError()
               << "the global-load path supports 256xf32 (4 f32 per PE) or "
                  "512xf32 (8 f32 per PE, one matrix slab); a dot's operand "
                  "buffers are not loaded here at all -- the block grid places "
                  "them itself";
      unsigned planes = perPe / 2;      // longwords per PE
      uint64_t planeWords = 64;         // one longword per PE, per plane
      uint64_t l1 = llvm::alignTo(nextL1, 64);
      nextL1 = l1 + planes * planeWords;
      uint64_t l2 = llvm::alignTo(nextL2, 64);
      nextL2 = l2 + planes * planeWords;
      unsigned arg = itb->second;
      Location lloc = load.getLoc();
      if (wroteBack) {
        // Drain the L2BM local port between a writeback and this load (see
        // `wroteBack`): two empty steps, which print as `nop/2`.
        mncore::FenceOp::create(body, lloc);
        mncore::FenceOp::create(body, lloc);
      }
      // The buffer is plane-ordered, so placement is one 64-longword copy per
      // plane; MV moves 64 longwords at a time (design doc 22.33).
      for (unsigned pl = 0; pl < planes; ++pl)
        mncore::DmaOp::create(body, lloc, /*src_space=*/1u,
                              /*src_addr=*/pl * planeWords,
                              l2 + pl * planeWords, l1 + pl * planeWords,
                              planeWords, nextTag++, 0u, /*src_arg=*/arg);
      // The MV/L2B transfers are asynchronous; the PE read must not share a
      // step with them. Two empty steps are required before the L1B read
      // (measured on assemble3's hazard checker: one is rejected).
      mncore::FenceOp::create(body, lloc);
      mncore::FenceOp::create(body, lloc);
      auto tokenType = mncore::TokenType::get(ctx);
      SmallVector<Value> tok;
      for (unsigned pl = 0; pl < planes; ++pl)
        tok.push_back(mncore::LbOpOp::create(body, lloc, tokenType, 0u,
                                             l1 + pl * planeWords, 1u, 0u));
      Value value;
      if (perPe == 4) {
        value = mncore::VecLoadOp::create(
            body, lloc, VectorType::get({4}, rt.getElementType()),
            mncore::L1Mode::Distributed, tok[0], tok[1]);
      } else {
        value = mncore::SlabLoadOp::create(
            body, lloc, VectorType::get({8}, rt.getElementType()),
            mncore::L1Mode::Distributed, tok[0], tok[1], tok[2], tok[3]);
      }
      map[load.getResult()] = value;
      continue;
    }

    if (auto store = dyn_cast<triton::StoreOp>(op)) {
      if (dotStores.contains(&op))
        continue;   // the dot branch wrote this buffer block by block
      if (store.getMask())
        return store.emitError()
               << "only unmasked full-tile stores are supported";
      Value baseArg;
      int64_t N = 0;
      if (!matchFullTile(store.getPtr(), baseArg, N))
        return store.emitError()
               << "tt.store must be a whole tile at `base_ptr` whose offsets "
                  "are exactly the row-major index 0..N-1 (1-D "
                  "`base_ptr + tl.arange(0, N)`, or a 2-D "
                  "`base_ptr + rm[:, None] * C + rn[None, :]`); dynamic or "
                  "partial addressing is rejected";
      auto itb = ptrArgs.find(baseArg);
      if (itb == ptrArgs.end())
        return store.emitError() << "store base is not a kernel buffer argument";
      auto val = map.find(store.getValue());
      if (val == map.end())
        return store.emitError() << "store value is not an mncore vector";
      unsigned perPe = (unsigned)(N / 64);
      if (perPe != 4 && perPe != 8)
        return store.emitError()
               << "the global-store path supports a 256xf32 tile (4 f32 per "
                  "PE) or a 512xf32 tile (8 f32 per PE, the matrix-role "
                  "width)";
      unsigned planes = perPe / 2;
      uint64_t planeWords = 64;
      uint64_t l1 = llvm::alignTo(nextL1, 64);
      nextL1 = l1 + planes * planeWords;
      uint64_t l2 = llvm::alignTo(nextL2, 64);
      nextL2 = l2 + planes * planeWords;
      unsigned arg = itb->second;
      Location sloc = store.getLoc();
      auto tokenType = mncore::TokenType::get(ctx);
      SmallVector<Value> tok;
      for (unsigned pl = 0; pl < planes; ++pl)
        tok.push_back(mncore::LbOpOp::create(body, sloc, tokenType, 0u,
                                             l1 + pl * planeWords, 1u, 0u));
      if (perPe == 4) {
        mncore::VecStoreOp::create(body, sloc, val->second,
                                   mncore::L1Mode::Distributed, tok[0], tok[1],
                                   0);
      } else {
        mncore::SlabStoreOp::create(body, sloc, val->second,
                                    mncore::L1Mode::Distributed, tok[0],
                                    tok[1], tok[2], tok[3], 0);
      }
      // Separate the PE collect from the L2B collect that reads L1BM.
      mncore::FenceOp::create(body, sloc);
      for (unsigned pl = 0; pl < planes; ++pl)
        mncore::DmaStoreOp::create(body, sloc, l1 + pl * planeWords,
                                   l2 + pl * planeWords,
                                   /*dr_addr=*/pl * planeWords, planeWords,
                                   nextTag++, /*l1b=*/0, /*dr_arg=*/arg);
      wroteBack = true;
      continue;
    }

    if (auto store = dyn_cast<ttg::LocalStoreOp>(op)) {
      auto it = allocs.find(store.getDst());
      if (it == allocs.end())
        return store.emitError()
               << "local_store destination is not an allocated L1BM buffer";
      auto val = map.find(store.getSrc());
      if (val == map.end())
        return store.emitError()
               << "local_store value is not an mncore vector";
      auto dist = analyzeDistribution(store.getSrc().getType());
      if (failed(dist))
        return store.emitError()
               << "local_store layout is neither a full 64-PE distribution "
                  "nor a broadcast";
      if (dist->kind == Distribution::Broadcast)
        return store.emitError()
               << "a broadcast store is not defined; use a full 64-PE "
                  "distribution";
      if (dist->kind == Distribution::MAB)
        return store.emitError()
               << "MAB store is not derivable from a Triton layout: l1bmm@"
                  "<mabadr> needs a physical MAB selector (0..15) that a "
                  "replicated layout does not determine, and one transfer "
                  "writes a single PE's double-longword operand, not the "
                  "4-PE MAB lane. Emit an explicit mncore.vec.store with the "
                  "selector instead";
      if (dist->kind == Distribution::MAB4)
        return store.emitError()
               << "MAB4 store is not expressible: l1bmm4@<k> writes exactly "
                  "one of the four MAB groups, the groups hold different data "
                  "(lane bit 4 / warp select the group), and "
                  "mncore.vec.store carries a single value token, so the four "
                  "per-group sources cannot be written";
      // Distributed writeback: l1bmd has no double-longword form, so the two
      // longword planes are separate collect transfers.
      auto tokenType = mncore::TokenType::get(ctx);
      Location loc = store.getLoc();
      uint64_t plane = it->second.longwords / 2;
      auto lo = mncore::LbOpOp::create(body, loc, tokenType, 0u,
                                       it->second.base, 1u, 0u);
      auto hi = mncore::LbOpOp::create(body, loc, tokenType, 0u,
                                       it->second.base + plane, 1u, 0u);
      mncore::VecStoreOp::create(body, loc, val->second,
                                 mncore::L1Mode::Distributed, lo, hi,
                                 /*set=*/0);
      continue;
    }

    return op.emitError()
           << "unsupported operation in tritongpu-to-mncore: "
           << op.getName().getStringRef();
  }

  fn.erase();
  return success();
}

struct TritonGPUToMNCorePass
    : public PassWrapper<TritonGPUToMNCorePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonGPUToMNCorePass)

  StringRef getArgument() const final { return "tritongpu-to-mncore"; }
  StringRef getDescription() const final {
    return "Lower a restricted TritonGPU kernel to the mncore dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mncore::MNCoreDialect, func::FuncDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto fn : llvm::make_early_inc_range(module.getOps<triton::FuncOp>()))
      if (failed(convertFunction(fn)))
        return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::mncore::createTritonGPUToMNCorePass() {
  return std::make_unique<TritonGPUToMNCorePass>();
}
