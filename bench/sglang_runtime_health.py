"""Recognize fatal GPU execution logs without assigning a hardware cause.

Controllers must retain the evidence and stop before restoring another GPU
runtime. An OOM or a missing tuning configuration alone is not this fault class.
"""

_FATAL_GPU_MARKERS = (
    "memory access fault",
    "illegal memory access",
    "device-side assert",
    "hsa_status_error_exception",
    "hiperrorillegaladdress",
    "gpu coredump:",
    "hw exception by gpu",
    "reason :gpu hang",
    "hiperrorlaunchfailure",
    "hiperrorassert",
)

# The container log is not the first place a failed GPU always appears.  In
# particular, an amdgpu MES timeout can precede a ROCr queue exception by
# minutes.  A correctable PCIe AER event is kept separate: it is evidence to
# investigate, not proof of a GPU execution fault or its cause.
_FATAL_KERNEL_MARKERS = (
    "failed to respond to msg=",
    "mes might be in unrecoverable state",
    "gpu reset begin",
    "gpu reset failed",
    "gpu fault detected",
    "[gfxhub] page fault",
    "[mmhub] page fault",
)
_PCIE_WARNING_MARKERS = ("badtlp", "badllp", "replay timer timeout", "[12] timeout")


def gpu_fault_lines(text):
    return [line for line in text.splitlines()
            if any(marker in line.lower() for marker in _FATAL_GPU_MARKERS)]


def kernel_gpu_fault_lines(text):
    """Return kernel messages that require stopping GPU-runtime reuse.

    The caller must scope journal input to the *current boot* and participating
    GPUs.  This function classifies an observation, not its root cause.
    """
    return [line for line in text.splitlines()
            if "amdgpu" in line.lower()
            and any(marker in line.lower() for marker in _FATAL_KERNEL_MARKERS)]


def pcie_warning_lines(text):
    """Return AER symptoms for review; these alone are not fatal GPU faults."""
    return [line for line in text.splitlines()
            if "pcieport" in line.lower()
            and any(marker in line.lower() for marker in _PCIE_WARNING_MARKERS)]
