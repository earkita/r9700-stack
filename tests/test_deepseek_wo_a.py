"""CPU contract tests in the pinned DeepSeek image; no GPU access required."""
import os
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.model_executor.kernels.linear.mxfp8.emulation import EmulationMxfp8LinearKernel
from vllm.model_executor.layers.quantization.quark.schemes import QuarkOCP_MX
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Static, kMxfp8Dynamic
from r9700_vllm.compat.deepseek_wo_a import quark_wo_a_input, install


def layer():
    scheme = QuarkOCP_MX(kMxfp8Static, kMxfp8Dynamic, scale_block_rows=32)
    scheme.ocp_mx_linear = object.__new__(EmulationMxfp8LinearKernel)
    return SimpleNamespace(scheme=scheme)


@pytest.mark.parametrize("rows", [1, 4, 32])
def test_real_group_boundaries_and_zero_blocks(rows):
    torch.manual_seed(41013)
    x = torch.randn(rows, 8, 512).bfloat16()
    x[..., :32] = 0
    x[..., 32:64] *= 128
    original = x.clone()
    blocks = x.float().reshape(rows, -1, 32)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(
        torch.finfo(torch.float32).tiny) / 448)).clamp(-127, 127))
    expected = ((blocks / scale[..., None]).to(torch.float8_e4m3fn).float()
                * scale[..., None]).reshape_as(x).bfloat16()
    actual = quark_wo_a_input(layer(), x)
    assert not torch.equal(original, expected)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(x, original, atol=0, rtol=0)


def test_native_and_non_quark_unchanged():
    x = torch.ones(1, 8, 512).bfloat16()
    assert quark_wo_a_input(SimpleNamespace(), x) is x
    native = layer()
    native.scheme.ocp_mx_linear = object()
    assert quark_wo_a_input(native, x) is x
    prequantized = SimpleNamespace(data=x, scale=None)
    assert quark_wo_a_input(layer(), prequantized) is prequantized


def test_invalid_geometry_and_empty():
    with pytest.raises(ValueError, match="BF16"):
        quark_wo_a_input(layer(), torch.ones(1, 8, 512))
    x = torch.empty(0, 8, 512).bfloat16()
    assert quark_wo_a_input(layer(), x) is x


def test_scoped_hook_idempotence_and_positions():
    with patch.object(torch.cuda, "get_device_properties",
                      return_value=SimpleNamespace(gcnArchName="gfx1201")):
        from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention as Attention
    def original(self, attn_out, positions):
        return attn_out, positions
    with patch.object(Attention, "_o_proj", original), patch.dict(os.environ, R9K_DEEPSEEK_WO_A_QDQ="1"):
        install()
        wrapper = Attention._o_proj
        install()
        assert Attention._o_proj is wrapper
        x = torch.linspace(-1.7, 2.3, 4096).reshape(1, 8, 512).bfloat16()
        positions = torch.tensor([123])
        y, pos = wrapper(SimpleNamespace(wo_a=layer()), x, positions)
        torch.testing.assert_close(y, quark_wo_a_input(layer(), x), atol=0, rtol=0)
        assert pos is positions


def test_opt_in_and_interface_guard():
    with patch.dict(os.environ, R9K_DEEPSEEK_WO_A_QDQ="0"):
        install()
    with patch.dict(os.environ, R9K_DEEPSEEK_WO_A_QDQ="invalid"), pytest.raises(ValueError):
        install()
    with patch.object(torch.cuda, "get_device_properties",
                      return_value=SimpleNamespace(gcnArchName="gfx1201")):
        from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention as Attention
    with patch.object(Attention, "_o_proj", lambda self, x: x), patch.dict(os.environ, R9K_DEEPSEEK_WO_A_QDQ="1"):
        with pytest.raises(RuntimeError, match="unsupported"):
            install()


@pytest.mark.skipif(not os.environ.get("DEEPSEEK_MODEL_CONFIG"), reason="Optional local checkpoint")
@pytest.mark.parametrize("layer_id", [0, 20, 39])
@pytest.mark.parametrize("rows", [1, 4, 32])
def test_checkpoint_projection_against_independent_qdq(layer_id, rows):
    from safetensors import safe_open
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import dequant_mxfp8_to_bf16
    with patch.object(torch.cuda, "get_device_properties",
                      return_value=SimpleNamespace(gcnArchName="gfx1201")):
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inv_rope_einsum
    model = Path(os.environ["DEEPSEEK_MODEL_CONFIG"]).parent
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    key = f"layers.{layer_id}.attn.wo_a"
    # One TP8 group only; all tensors stay on CPU.
    with safe_open(model / index[key + ".weight"], framework="pt", device="cpu") as reader:
        weight = reader.get_slice(key + ".weight")[:1024, :].contiguous()
    with safe_open(model / index[key + ".weight_scale"], framework="pt", device="cpu") as reader:
        scale = reader.get_slice(key + ".weight_scale")[:32, :].contiguous().view(torch.uint8)
    expanded = scale.repeat_interleave(32, dim=0)
    expected_weight = (weight.float() * torch.exp2(expanded.float() - 127).repeat_interleave(32, dim=1)).bfloat16()
    module = layer()
    module.weight = dequant_mxfp8_to_bf16(weight, expanded)
    module.weight_scale = expanded
    torch.testing.assert_close(module.weight, expected_weight, atol=0, rtol=0)
    torch.manual_seed(41013 + rows)
    x = torch.randn(rows, 8, 512).bfloat16()
    x[..., :32] = 0
    groups = x.float().reshape(rows, -1, 32)
    act_scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(
        torch.finfo(torch.float32).tiny) / 448)).clamp(-127, 127))
    ref_input = ((groups / act_scale[..., None]).to(torch.float8_e4m3fn).float()
                 * act_scale[..., None]).reshape(rows, 1, 4096).bfloat16()
    expected = torch.einsum("tgd,grd->tgr", ref_input, expected_weight.reshape(1, 1024, 4096))
    args = (torch.arange(rows), 64, 1, 1024, module)
    stock = rocm_inv_rope_einsum(None, x, *args, inverse_rope=False)
    candidate = rocm_inv_rope_einsum(None, quark_wo_a_input(module, x), *args, inverse_rope=False)
    assert not torch.equal(stock, expected), "Fixture must expose the missing activation QDQ"
    torch.testing.assert_close(candidate, expected, atol=0, rtol=0)
