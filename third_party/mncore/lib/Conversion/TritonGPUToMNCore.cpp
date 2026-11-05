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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

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
static FailureOr<DistributionInfo> analyzeDistribution(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || tensor.getRank() != 1)
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
  auto basesOf = [&](StringRef name) {
    SmallVector<int32_t> out;
    auto key = StringAttr::get(tensor.getContext(), name);
    auto it = ll.getBases().find(key);
    if (it == ll.getBases().end())
      return out;
    for (const auto &basis : it->second)
      out.push_back(basis[0]);
    return out;
  };
  auto reg = basesOf("register");
  auto lane = basesOf("lane");
  auto warp = basesOf("warp");

  // 4 f32 (2 f64 / 8 f16) per PE: exactly log2(lanes) register bits.
  if ((1u << reg.size()) != lanes)
    return failure();

  // 64 PE lanes = 32 lane bits + 1 warp bit.
  SmallVector<int32_t> bits;
  bits.append(lane.begin(), lane.end());
  bits.append(warp.begin(), warp.end());
  if (bits.size() > 6)
    return failure();
  // Canonicalized replicated layouts may drop the zero lane/warp bits.
  bits.resize(6, 0);
  unsigned nonzero = 0;
  for (int32_t b : bits)
    if (b != 0)
      ++nonzero;

  auto isZero = [&](unsigned i) { return bits[i] == 0; };
  int64_t elements = tensor.getShape()[0];
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

// Match `tt.addptr(tt.splat(ptr_arg), tt.make_range(0, N))`, i.e. a whole
// contiguous tile starting at a kernel pointer argument. Only compile-time
// affine offsets are accepted; anything else must be rejected by the caller.
static bool matchFullTile(Value ptr, Value &baseArg, int64_t &N) {
  auto add = ptr.getDefiningOp<triton::AddPtrOp>();
  if (!add)
    return false;
  auto splat = add.getPtr().getDefiningOp<triton::SplatOp>();
  if (!splat)
    return false;
  auto range = add.getOffset().getDefiningOp<triton::MakeRangeOp>();
  if (!range || range.getStart() != 0)
    return false;
  N = range.getEnd();
  if (N <= 0)
    return false;
  baseArg = splat.getSrc();
  return isa<BlockArgument>(baseArg);
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
// B per 8-row M block, chaining K in blocks of 4 through y. Nothing of that
// sequence exists yet, and the TTGIR a `tl.dot` produces today cannot feed it:
// with a blocked parent, `fmaDotToLinearLayout` puts the whole K dimension in
// every thread's registers (32 f32 per thread for a 32x32x32 dot), while the
// MAU consumes K one element per PE. Every dot is rejected; the function only
// chooses which clause of the contract to cite, so the diagnostic stays
// actionable instead of degrading to "unsupported operation".
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
         << " is not lowered yet; the MAU GEMM sequence is pending. A block "
            "is: mwrite the 8x4 A tile into a matrix face, then one fmfma per "
            "4 columns of B with x at word width (4 f32, one per PE) and "
            "y/dst at longword width (8 f32, two per PE), chaining K in "
            "blocks of 4 through y (mncore_intrinsics_design.md section 22). "
            "This dot carries A: "
         << aLayout << ", B: " << bLayout
         << "; the blocked-parent dot-operand layout replicates the whole K "
            "dimension into every thread's registers, whereas the MAU "
            "consumes K one element per PE of a MAB, so the operands must be "
            "given an mncore dot layout first";
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

  // A dot is rejected before the linear walk: the contract is a property of
  // the whole kernel (operand layout, K distribution), so citing it is more
  // useful than letting the walk fail on some earlier op.
  for (Operation &op : fn.getBody().front())
    if (auto dot = dyn_cast<triton::DotOp>(op))
      return rejectDot(dot);

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
             << "; expected a 1-D blocked tensor with 4xf32 / 2xf64 / 8xf16 "
                "per PE, or an *fp32 buffer";
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
                  "(threads*warps == 64, elements == 64*lanes) nor a "
                  "broadcast (one thread, elements == lanes)";
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
        return add.emitError() << "addf operands are not mncore vectors";
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

    // Pointer expressions are consumed by the full-tile load/store patterns.
    if (isa<triton::MakeRangeOp, triton::SplatOp, triton::AddPtrOp>(op))
      continue;

    if (auto load = dyn_cast<triton::LoadOp>(op)) {
      if (load.getMask() || load.getOther())
        return load.emitError()
               << "only unmasked full-tile loads are supported";
      Value baseArg;
      int64_t N = 0;
      if (!matchFullTile(load.getPtr(), baseArg, N))
        return load.emitError()
               << "tt.load must be `base_ptr + tl.arange(0, N)` over a whole "
                  "tile; dynamic or partial addressing is rejected";
      auto itb = ptrArgs.find(baseArg);
      if (itb == ptrArgs.end())
        return load.emitError() << "load base is not a kernel buffer argument";
      auto rt = dyn_cast<RankedTensorType>(load.getType());
      if (!rt || !rt.getElementType().isF32() || N != 256)
        return load.emitError()
               << "the first global-load path supports a 256xf32 tile "
                  "(64 PEs x 4 f32)";
      uint64_t planeWords = N / 4; // 64 longwords per plane
      uint64_t l1 = llvm::alignTo(nextL1, 64);
      nextL1 = l1 + 2 * planeWords;
      uint64_t l2 = llvm::alignTo(nextL2, 64);
      nextL2 = l2 + 2 * planeWords;
      unsigned arg = itb->second;
      Location lloc = load.getLoc();
      mncore::DmaOp::create(body, lloc, /*src_space=*/1u, /*src_addr=*/0, l2,
                            l1, planeWords, nextTag++, 0u,
                            /*src_arg=*/arg);
      mncore::DmaOp::create(body, lloc, /*src_space=*/1u,
                            /*src_addr=*/planeWords, l2 + planeWords,
                            l1 + planeWords, planeWords, nextTag++, 0u,
                            /*src_arg=*/arg);
      // The MV/L2B transfers are asynchronous; the PE read must not share a
      // step with them. Two empty steps are required before the L1B read
      // (measured on assemble3's hazard checker: one is rejected).
      mncore::FenceOp::create(body, lloc);
      mncore::FenceOp::create(body, lloc);
      auto tokenType = mncore::TokenType::get(ctx);
      auto lo = mncore::LbOpOp::create(body, lloc, tokenType, 0u, l1, 1u, 0u);
      auto hi = mncore::LbOpOp::create(body, lloc, tokenType, 0u,
                                       l1 + planeWords, 1u, 0u);
      auto value = mncore::VecLoadOp::create(
          body, lloc, VectorType::get({4}, rt.getElementType()),
          mncore::L1Mode::Distributed, lo, hi);
      map[load.getResult()] = value;
      continue;
    }

    if (auto store = dyn_cast<triton::StoreOp>(op)) {
      if (store.getMask())
        return store.emitError()
               << "only unmasked full-tile stores are supported";
      Value baseArg;
      int64_t N = 0;
      if (!matchFullTile(store.getPtr(), baseArg, N))
        return store.emitError()
               << "tt.store must be `base_ptr + tl.arange(0, N)` over a whole "
                  "tile; dynamic or partial addressing is rejected";
      auto itb = ptrArgs.find(baseArg);
      if (itb == ptrArgs.end())
        return store.emitError() << "store base is not a kernel buffer argument";
      auto val = map.find(store.getValue());
      if (val == map.end())
        return store.emitError() << "store value is not an mncore vector";
      if (N != 256)
        return store.emitError()
               << "the first global-store path supports a 256xf32 tile";
      uint64_t planeWords = N / 4;
      uint64_t l1 = llvm::alignTo(nextL1, 64);
      nextL1 = l1 + 2 * planeWords;
      uint64_t l2 = llvm::alignTo(nextL2, 64);
      nextL2 = l2 + 2 * planeWords;
      unsigned arg = itb->second;
      Location sloc = store.getLoc();
      auto tokenType = mncore::TokenType::get(ctx);
      auto lo = mncore::LbOpOp::create(body, sloc, tokenType, 0u, l1, 1u, 0u);
      auto hi = mncore::LbOpOp::create(body, sloc, tokenType, 0u,
                                       l1 + planeWords, 1u, 0u);
      mncore::VecStoreOp::create(body, sloc, val->second,
                                 mncore::L1Mode::Distributed, lo, hi, 0);
      // Separate the PE collect from the L2B collect that reads L1BM.
      mncore::FenceOp::create(body, sloc);
      mncore::DmaStoreOp::create(body, sloc, l1, l2, /*dr_addr=*/0,
                                 planeWords, nextTag++, /*l1b=*/0,
                                 /*dr_arg=*/arg);
      mncore::DmaStoreOp::create(body, sloc, l1 + planeWords,
                                 l2 + planeWords, /*dr_addr=*/planeWords,
                                 planeWords, nextTag++, /*l1b=*/0,
                                 /*dr_arg=*/arg);
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
