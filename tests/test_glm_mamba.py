"""Pinned V2 request initialization, including resumed prefix state on GPU."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from r9700_vllm.compat.glm_mamba import _wrap_add_request, _wrap_group_info


def state(model="glm5_next", align=True, device="cpu", attention=640, mamba=16):
    s = object.__new__(MambaHybridModelState)
    s.model_config = NS(hf_config=NS(model_type=model))
    s.cache_config = NS(block_size=attention, mamba_block_size=mamba)
    s._align_mode = align
    s._mamba_spec = None
    s.rope_state = None
    s.prompt_embeds_state = Mock()
    s.num_accepted_tokens_gpu = torch.full((1,), 9, device=device, dtype=torch.int32)
    s._mamba_state_idx_gpu = torch.full((1,), -99, device=device, dtype=torch.int32)
    return s


@pytest.mark.parametrize("tokens,expected", [(0, -1), (1, 0), (16, 0), (17, 1), (640, 39), (7680, 479)])
def test_request_seed_and_upstream_side_effects(tokens, expected):
    s, req = state(), NS(num_computed_tokens=tokens)
    _wrap_add_request(MambaHybridModelState.add_request)(s, 0, req)
    assert s._mamba_state_idx_gpu.item() == expected
    assert s.num_accepted_tokens_gpu.item() == 1
    s.prompt_embeds_state.add_request.assert_called_once_with(0, req)


@pytest.mark.parametrize("model,align,expected", [("other", True, 11), ("glm5_next", False, -99)])
def test_scope(model, align, expected):
    s = state(model=model, align=align)
    _wrap_add_request(MambaHybridModelState.add_request)(s, 0, NS(num_computed_tokens=7680))
    assert s._mamba_state_idx_gpu.item() == expected


def test_effective_spec_guard():
    s = state()
    log = Mock()
    _wrap_group_info(lambda s, cfg: ([2], NS(block_size=16)), log)(s, None)
    log.info.assert_called_once()
    with pytest.raises(RuntimeError, match="different block sizes"):
        _wrap_group_info(lambda s, cfg: ([2], NS(block_size=640)), log)(s, None)


@pytest.mark.parametrize("attention,mamba,tokens", [(640, 16, 7680), (64, 7168, 14336)])
def test_gpu_precopy_uses_resumed_mamba_column(attention, mamba, tokens):
    from vllm.v1.worker.mamba_utils import preprocess_mamba_align_fused_kernel
    s = state(device="cuda", attention=attention, mamba=mamba)
    req = NS(num_computed_tokens=tokens)
    original = MambaHybridModelState.add_request
    original(s, 0, req)
    wrong_seed = s._mamba_state_idx_gpu.item()
    _wrap_add_request(original)(s, 0, req)
    src, off = torch.empty_like(s._mamba_state_idx_gpu), torch.empty_like(s._mamba_state_idx_gpu)
    tensor = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)
    preprocess_mamba_align_fused_kernel[(1,)](
        tensor([0]), s._mamba_state_idx_gpu, tensor([tokens]), tensor([0, 1]),
        s.num_accepted_tokens_gpu, src, off, 1, BLOCK_SIZE=256, MAMBA_BLOCK_SIZE=mamba)
    assert src.item() == tokens // mamba - 1
    assert src.item() != wrong_seed
    assert s._mamba_state_idx_gpu.item() == tokens // mamba
    assert off.item() == 0
