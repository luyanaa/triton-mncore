include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# MN-Core Triton plugin.
#
# Requires LLVM_SYSPATH to point at an install/build tree of llvm-mncore with
# -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=MNCore and MLIR enabled, so that
# libMLIRMNCore{ Dialect,ToLLVM }.a and the MNCore target are available.

if(TRITON_BUILD_PYTHON_MODULE)
  find_library(MNCORE_DIALECT_LIB NAMES MLIRMNCoreDialect
    PATHS "${LLVM_LIBRARY_DIR}" NO_DEFAULT_PATH)
  find_library(MNCORE_TO_LLVM_LIB NAMES MLIRMNCoreToLLVM
    PATHS "${LLVM_LIBRARY_DIR}" NO_DEFAULT_PATH)
  if(NOT MNCORE_DIALECT_LIB OR NOT MNCORE_TO_LLVM_LIB)
    message(FATAL_ERROR
      "MN-Core MLIR libraries not found in ${LLVM_LIBRARY_DIR}. "
      "Build/install llvm-mncore with MLIR enabled and the MNCore target.")
  endif()

  # libMLIRMNCoreToLLVM references these conversion/dialect libraries, and
  # Triton's final link does not include all of them.
  set(MNCORE_MLIR_DEPS "")
  foreach(_lib MLIRArithToLLVM MLIRFuncToLLVM MLIRArithDialect
               MLIRFuncDialect MLIRControlFlowToLLVM
               MLIRArithAttrToLLVMConversion MLIRArithCommonConversion
               MLIRAnalysisBase MLIRDataLayoutInterfaces
               MLIRLLVMCommonConversion MLIRTransformUtils MLIRRewrite)
    find_library(_mncore_${_lib} NAMES ${_lib}
      PATHS "${LLVM_LIBRARY_DIR}" NO_DEFAULT_PATH)
    if(NOT _mncore_${_lib})
      message(FATAL_ERROR "Missing ${_lib} in ${LLVM_LIBRARY_DIR}")
    endif()
    list(APPEND MNCORE_MLIR_DEPS ${_mncore_${_lib}})
  endforeach()

  add_triton_library(TritonMNCoreConversion
    lib/Conversion/TritonGPUToMNCore.cpp
    LINK_LIBS PUBLIC
    TritonIR
    TritonGPUIR
    ${MNCORE_DIALECT_LIB}
    ${MNCORE_MLIR_DEPS})

  add_triton_plugin(TritonMncore triton_mncore.cc
    LINK_LIBS PUBLIC
    TritonMNCoreConversion
    ${MNCORE_TO_LLVM_LIB}
    ${MNCORE_MLIR_DEPS})
  target_link_libraries(TritonMncore PRIVATE Python3::Module nanobind-static)
endif()
