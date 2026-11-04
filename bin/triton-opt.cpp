#include "./RegisterTritonDialects.h"

#include "mncore/Conversion/Passes.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerTritonDialects(registry);
  mlir::registerPass(
      []() { return mlir::mncore::createTritonGPUToMNCorePass(); });

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton (GPU) optimizer driver\n", registry));
}
