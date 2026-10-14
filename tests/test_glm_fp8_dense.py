"""CPU dispatch gates: scope, fallback, shape, bias, and opt-in."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
torch = pytest.importorskip("torch")
from r9700_vllm.compat.glm_fp8_dense import wrap_apply, patch


def test_disabled_does_not_import_vllm(monkeypatch):
    monkeypatch.setenv("R9K_GLM_DENSE_FP8", "0")
    assert patch() is False
    monkeypatch.setenv("R9K_GLM_DENSE_FP8", "invalid")
    with pytest.raises(ValueError):
        patch()


def layer():
    return NS(weight=torch.empty(3072, 4096, device="meta"), weight_scale=object(),
              _r9k_glm_dense_packed=torch.empty(1))


@pytest.mark.parametrize("rows", [1, 5, 10, 20, 32])
def test_decode_bias_and_packed_weights(rows):
    original, gemm = Mock(), Mock(return_value=torch.ones(rows, 3072))
    obj = layer()
    x = torch.zeros(rows, 4096, dtype=torch.bfloat16)
    bias = torch.ones(3072)
    out = wrap_apply(original, gemm)(object(), obj, x, bias)
    assert torch.equal(out, torch.full((rows, 3072), 2.))
    original.assert_not_called()
    assert gemm.call_args.args[1] is obj._r9k_glm_dense_packed
    assert gemm.call_args.args[2] is obj.weight_scale


@pytest.mark.parametrize("x", [torch.empty(0, 4096), torch.empty(33, 4096),
                              torch.empty(5, 4096), torch.empty(5, 4095, dtype=torch.bfloat16),
                              torch.empty(5, 8192, dtype=torch.bfloat16)[:, ::2]])
def test_unsupported_input_uses_original(x):
    original, gemm = Mock(), Mock()
    obj = layer()
    result = wrap_apply(original, gemm)(object(), obj, x)
    assert result is original.return_value
    gemm.assert_not_called()


def test_unmarked_layer_always_uses_original():
    original, gemm = Mock(), Mock()
    obj = layer()
    del obj._r9k_glm_dense_packed
    wrap_apply(original, gemm)(object(), obj, torch.zeros(5,4096,dtype=torch.bfloat16))
    original.assert_called_once()
    gemm.assert_not_called()


def test_runtime_names_and_nested_text_config():
    from r9700_vllm.compat.glm_fp8_dense import target_model, target_prefix
    for model_type in ("glm5_next", "glm5_next_text"):
        cfg = NS(model_config=NS(hf_config=NS(model_type=model_type)),
                 parallel_config=NS(tensor_parallel_size=8))
        assert target_model(cfg)
        cfg.parallel_config.tensor_parallel_size = 4
        assert not target_model(cfg)
    cfg.model_config.hf_config.model_type = "qwen4_exp"
    cfg.parallel_config.tensor_parallel_size = 8
    assert not target_model(cfg)
    assert target_prefix("language_model.model.layers.3.self_attn.q_b_proj")
    assert target_prefix("model.language_model.layers.3.self_attn.q_b_proj")
    for name in ("visual.blocks.3.proj", "draft.model.layers.45.fc", "model.layers.3.mlp"):
        assert not target_prefix(name)


def test_prepare_preserves_checkpoint_and_rejects_other_layouts():
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym, kFp8Static128BlockSym,
    )
    from r9700_vllm.compat.glm_fp8_dense import prepare
    backend = type("TritonFp8BlockScaledMMKernel", (), {})()
    backend.quant_fp8 = NS(use_aiter=False)
    scheme = NS(_r9k_glm_dense=True, activation_quant_key=kFp8Dynamic128Sym,
                weight_quant_key=kFp8Static128BlockSym, weight_scale_dtype=torch.float32,
                input_dtype=torch.bfloat16, out_dtype=torch.bfloat16, fp8_linear=backend)
    weight = NS(shape=(3072, 4096), dtype=torch.float8_e4m3fn, is_cuda=True,
                stride=lambda dim: (4352, 1)[dim])
    scales = NS(shape=(24, 32), dtype=torch.float32, is_contiguous=lambda: True)
    obj = NS(weight=weight, weight_scale=scales, register_buffer=Mock())
    pack = Mock()
    assert prepare(scheme, obj, pack)
    pack.assert_called_once_with(weight)
    obj.register_buffer.assert_called_once_with("_r9k_glm_dense_packed", pack.return_value, persistent=False)
    assert obj.weight is weight and obj.weight_scale is scales
    for field, value in (("_r9k_glm_dense", False), ("weight_scale_dtype", torch.uint8),
                         ("input_dtype", torch.float16), ("out_dtype", torch.float32)):
        before = getattr(scheme, field)
        setattr(scheme, field, value)
        pack.reset_mock()
        assert not prepare(scheme, obj, pack)
        pack.assert_not_called()
        setattr(scheme, field, before)
    backend.quant_fp8.use_aiter = True
    assert not prepare(scheme, obj, pack)


def test_padding_preserves_fp8_bytes():
    from r9700_vllm.kernels.fp8 import permute_fp8
    # ROCm pads rows by 256 bytes, then exposes the original logical shape.
    w = torch.randint(0, 255, (32, 512), dtype=torch.uint8)[:, :256]
    assert not w.is_contiguous()
    assert torch.equal(permute_fp8(w), permute_fp8(w.contiguous()))
