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
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace ttg = mlir::triton::gpu;

namespace {

enum class Distribution { Broadcast, Distributed };

struct DistributionInfo {
  Distribution kind;
  VectorType vectorType;
};

// MN-Core executes one expression across all 64 PEs (16 MABs x 4 PEs).
// A TTGIR tensor therefore maps to a per-PE 2-longword slice only when its
// blocked layout covers exactly 64 threads and 64 * lanes elements
// (Distributed), or when a single thread owns one vector and every PE sees
// the same value (Broadcast). Anything else is an incomplete distribution and
// is rejected rather than silently replicated.
static FailureOr<DistributionInfo> analyzeDistribution(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || tensor.getRank() != 1)
    return failure();
  auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(tensor.getEncoding());
  if (!blocked)
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

  auto contig = ttg::getContigPerThread(tensor);
  auto order = ttg::getOrder(tensor);
  if (contig.size() != 1 || contig[0] != lanes || order.size() != 1 ||
      order[0] != 0)
    return failure();

  uint64_t threads = 1;
  for (unsigned t : ttg::getThreadsPerWarp(tensor))
    threads *= t;
  for (unsigned w : blocked.getWarpsPerCTA())
    threads *= w;
  int64_t elements = tensor.getShape()[0];
  VectorType vectorType = VectorType::get({lanes}, element);

  if (threads == 64 && elements == 64LL * lanes)
    return DistributionInfo{Distribution::Distributed, vectorType};
  // A single-thread's-worth tensor is replicated to every PE, regardless of
  // how many threads the Triton layout names: every PE executes the same
  // expression and must see the same vector.
  if (elements == static_cast<int64_t>(lanes))
    return DistributionInfo{Distribution::Broadcast, vectorType};
  return failure();
}

static FailureOr<VectorType> convertTensorType(Type type) {
  auto info = analyzeDistribution(type);
  if (failed(info))
    return failure();
  return info->vectorType;
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

  SmallVector<Type> args;
  for (Type t : fn.getFunctionType().getInputs()) {
    auto vec = convertTensorType(t);
    if (failed(vec))
      return fn.emitError()
             << "unsupported kernel argument type " << t
             << "; expected a 1-D blocked tensor with 4xf32 / 2xf64 / 8xf16 "
                "contiguous per PE and order [0]";
    args.push_back(*vec);
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
      if (dist->kind == Distribution::Broadcast) {
        // l1bmp: one 2-longword broadcast.
        auto lb = mncore::LbOpOp::create(body, loc, tokenType, 0u,
                                         it->second.base, 2u, 0u);
        auto none = mncore::NoneOp::create(body, loc, tokenType);
        value = mncore::VecLoadOp::create(body, loc, dist->vectorType,
                                          mncore::L1Mode::Broadcast, lb, none);
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
