"""Numerical contract for the two upstream MXFP8 storage policies."""
import os
import pytest

if os.environ.get("R9700_DEEPSEEK_GPU_TEST") != "1":
    pytest.skip("Explicit GPU test opt-in required", allow_module_level=True)

import torch
import vllm.envs as envs
from vllm.model_executor.kernels.linear.mxfp8.emulation import EmulationMxfp8LinearKernel
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import MXFP8_SCALE_DTYPE


@pytest.mark.parametrize("shape", [(640, 5120), (5120, 15360)])
def test_load_time_and_per_call_dequant_match(shape, monkeypatch):
    torch.manual_seed(41005)
    n, k = shape
    weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.randint(120, 130, (n, k // 32), device="cuda",
                           dtype=torch.uint8).view(MXFP8_SCALE_DTYPE)
    kernel = object.__new__(EmulationMxfp8LinearKernel)
    layers = []
    for at_load in (True, False):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scales.clone(), requires_grad=False)
        monkeypatch.setattr(envs, "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD", at_load)
        kernel.process_weights_after_loading(layer)
        layers.append(layer)
    assert layers[0].weight.element_size() == 2
    assert layers[1].weight.element_size() == 1
    for rows in (1, 4, 20, 24, 512):
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        reference = kernel.apply_weights(layers[0], x)
        candidate = kernel.apply_weights(layers[1], x)
        assert torch.isfinite(candidate).all()
        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
