//===- triton_mncore.cc - Triton plugin for the MN-Core backend ------------===//
//
// Exposes the mncore dialect and the mncore -> LLVM conversion to Triton's
// Python pass manager, and initializes the MNCore LLVM target (Triton's own
// llvm.init_targets() only knows NVPTX/AMDGPU).
//
// Build with -DTRITON_CODEGEN_BACKENDS=mncore and LLVM_SYSPATH pointing at an
// install of /Users/yanlu/Documents/llvm-mncore (see README.md).
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MNCoreToLLVM/MNCoreToLLVM.h"
#include "mncore/Conversion/Passes.h"
#include "mlir/Dialect/MNCore/IR/MNCoreOps.h"
#include "mlir/Pass/PassManager.h"

#include "passes.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <stdexcept>

namespace py = nanobind;

// The MNCore target initialization functions are exported by the LLVM target
// but are not listed in llvm/Support/TargetSelect.h (that header only lists
// the standard targets).
extern "C" {
void LLVMInitializeMNCoreTargetInfo();
void LLVMInitializeMNCoreTarget();
void LLVMInitializeMNCoreTargetMC();
void LLVMInitializeMNCoreAsmPrinter();
}

void init_triton_mncore(py::module_ &m) {
  m.def("load_dialects", [](mlir::MLIRContext &ctx) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::mncore::MNCoreDialect>();
    ctx.appendDialectRegistry(registry);
  });

  m.def("init_targets", []() {
    LLVMInitializeMNCoreTargetInfo();
    LLVMInitializeMNCoreTarget();
    LLVMInitializeMNCoreTargetMC();
    LLVMInitializeMNCoreAsmPrinter();
  });

  ADD_PASS_WRAPPER_0("add_convert_mncore_to_llvm",
                     mlir::createConvertMNCoreToLLVMPass);

  m.def("add_tritongpu_to_mncore", [](mlir::PassManager &pm) {
    pm.addPass(mlir::mncore::createTritonGPUToMNCorePass());
  });
}
