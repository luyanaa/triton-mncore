"""MN-Core driver skeleton.

The device runtime (loader, buffer binding, relocation, launch) is owned by the
device-integration work line described in
llvm/lib/Target/MNCore/docs/mncore_kernel_abi_v1.md. This module only exposes
the target description Triton needs and fails loudly for the unimplemented
launch path.
"""

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase, Benchmarker


class MncoreDriver(DriverBase):

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def is_active(cls):
        # There is no in-process device discovery yet. Selecting this backend
        # explicitly is the only supported mode until mnrt is wired.
        return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty

    def get_current_target(self):
        return GPUTarget("mncore", "mncore2", 1)

    def get_active_torch_device(self):
        raise NotImplementedError(
            "MN-Core has no torch device integration yet; the runtime work line "
            "must provide a PrivateUse1 device and buffer handles")

    def get_benchmarker(self) -> Benchmarker:
        raise NotImplementedError(
            "MN-Core benchmarking needs an event/timing source from the runtime")

    def make_launcher(self, *args, **kwargs):
        raise NotImplementedError(
            "MN-Core launch is provided by mnrt (see docs/mncore_kernel_abi_v1.md)")

    def launch(self, *args, **kwargs):
        raise NotImplementedError(
            "MN-Core launch is provided by mnrt (see docs/mncore_kernel_abi_v1.md)")
