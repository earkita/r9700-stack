"""CPU regression for the actual HSA failure missed by the old controller."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "runtime_health", Path(__file__).parents[1] / "bench/sglang_runtime_health.py")
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)


def test_hsa_queue_exception_blocks_automatic_runtime_reuse():
    lines = [
        "Warning: Queue error: HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception.",
        "GPU coredump: Pipe handler '/usr/share/apport/apport' not found or not executable.",
    ]
    assert health.gpu_fault_lines("\n".join(lines)) == lines


def test_existing_memory_fault_signatures_remain_fatal():
    lines = ["Memory access fault by GPU", "HIP error: illegal memory access",
             "RuntimeError: device-side assert triggered", "hipErrorIllegalAddress"]
    assert health.gpu_fault_lines("\n".join(lines)) == lines


def test_actual_gpu_hang_signature_blocks_runtime_reuse():
    lines = [
        "HW Exception by GPU node-8 (Agent handle: 0x1ed61b00) reason :GPU Hang",
        "HW Exception by GPU node-8 (Agent handle: 0x4700f0f0) reason :GPU Hang",
    ]
    assert health.gpu_fault_lines("\n".join(lines)) == lines


def test_other_hip_execution_failures_block_runtime_reuse():
    lines = ["HIP error: hipErrorLaunchFailure", "HIP error: hipErrorAssert"]
    assert health.gpu_fault_lines("\n".join(lines)) == lines


def test_tuning_warning_or_oom_does_not_claim_memory_access_fault():
    text = ("Using default W8A8 Block FP8 kernel config.\n"
            "torch.OutOfMemoryError: tried to allocate 192 MiB\n"
            "No module named vllm._deepselect_C")
    assert health.gpu_fault_lines(text) == []


def test_kernel_mes_timeout_precedes_container_hsa_and_blocks_reuse():
    mes = "Oct 16 21:25:17 airig kernel: amdgpu 0000:c3:00.0: MES(1) failed to respond to msg=INVALIDATE_TLBS"
    hsa = "Oct 16 21:29:08 Warning: Queue error: HSA_STATUS_ERROR_EXCEPTION"
    assert health.kernel_gpu_fault_lines(mes + "\n" + hsa) == [mes]
    assert health.gpu_fault_lines(mes + "\n" + hsa) == [hsa]


def test_correctable_aer_is_separate_warning_not_inferred_gpu_fault():
    aer = "Oct 16 21:24:54 airig kernel: pcieport 0000:c7:00.0:    [ 6] BadTLP"
    timeout = "Oct 16 21:28:40 airig kernel: pcieport 0000:ca:00.0:    [12] Timeout"
    assert health.pcie_warning_lines(aer + "\n" + timeout) == [aer, timeout]
    assert health.kernel_gpu_fault_lines(aer + "\n" + timeout) == []
