//===- Passes.h - MN-Core Triton conversion passes --------------*- C++ -*-===//
#ifndef MNCORE_TRITON_CONVERSION_PASSES_H
#define MNCORE_TRITON_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;

namespace mncore {

/// Converts a TritonGPU function into the mncore dialect:
///   * ttg.local_alloc -> mncore.lbop (static L1BM slot)
///   * ttg.local_load  -> mncore.vec.load
///   * elementwise arith on tensors -> mncore.vec.*
///   * tt.func / tt.return -> func.func / func.return
///
/// The pass is deliberately strict: it accepts only straight-line bodies and
/// tensors whose layout gives each PE one 2-longword slice
/// (getContigPerThread == 4 for f32, 2 for f64, 8 for f16, order [0]).
std::unique_ptr<Pass> createTritonGPUToMNCorePass();

} // namespace mncore
} // namespace mlir

#endif // MNCORE_TRITON_CONVERSION_PASSES_H
