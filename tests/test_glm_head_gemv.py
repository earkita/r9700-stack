"""FP32 projection, pool rankings and graph replay on real GLM head weights."""
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from r9700_vllm.compat.glm_head_gemv import _wrap_forward
from r9700_vllm.router import router_gemm


def original_forward(self, hidden_states):
    return torch.mm(hidden_states.float(), self._wp_fp32)


def test_adapter_scopes_function_and_preserves_fallback():
    w = torch.randn(160, 4096).bfloat16()
    indexer = NS(wk_weights_proj=NS(weight=w), head_dim=128, _wp_fp32=w[128:].T.contiguous().float())
    gemv = Mock(return_value="candidate")
    adapted = _wrap_forward(original_forward, gemv)
    assert adapted(indexer, torch.ones(1, 4096).bfloat16()) == "candidate"
    assert adapted(indexer, torch.ones(8, 4096).bfloat16()) == "candidate"
    for x in (torch.ones(9, 4096).bfloat16(), torch.ones(1, 4096)):
        torch.testing.assert_close(adapted(indexer, x), original_forward(indexer, x), rtol=0, atol=0)
    assert gemv.call_count == 2
    assert original_forward.__globals__["torch"] is torch


def checkpoint_weights():
    from safetensors import safe_open
    path = Path(os.environ.get("GLM_CHECKPOINT", "/models/GLM-5.3-Flash-Quark-MXFP4"))
    if not path.exists():
        pytest.skip("Set GLM_CHECKPOINT to test the actual local checkpoint")
    weights = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    for name, shard in sorted(weights.items()):
        if name.endswith(".self_attn.indexer.weights_proj.weight") and ".layers." in name:
            with safe_open(path / shard, framework="pt", device="cpu") as f:
                yield name, f.get_tensor(name).cuda()


@pytest.mark.parametrize("rows", [1, 2, 5, 8])
def test_real_weights_fp64_reference_rankings_and_graph(rows):
    torch.manual_seed(933)
    count = 0
    for name, w in checkpoint_weights():
        if w.shape != (32, 4096):
            continue
        assert w.dtype == torch.bfloat16
        count += 1
        x = torch.randn(rows, 4096, device="cuda").bfloat16()
        original = x.float() @ w.T.contiguous().float()
        actual = router_gemm(x, w, out_bf16=False, split=8)
        ref = x.double() @ w.double().T
        original_error = (original.double() - ref).abs().max().item()
        actual_error = (actual.double() - ref).abs().max().item()
        assert actual_error <= 2 * original_error + 1e-6, (name, original_error, actual_error)
        assert actual.dtype == torch.float32
        # Same pooled-query scores, varying only head-projection weights.
        qk = torch.randn(32, 8192, device="cuda").relu()
        stock_scores = original @ qk
        actual_scores = actual @ qk
        score_ref = ref @ qk.double()
        stock_err = (stock_scores.double() - score_ref).abs().max()
        actual_err = (actual_scores.double() - score_ref).abs().max()
        assert actual_err <= 2 * stock_err + 2e-6, (name, stock_err, actual_err)
        for a, b in zip(stock_scores.topk(512).indices, actual_scores.topk(512).indices):
            assert set(a.tolist()) == set(b.tolist()), name
        for _ in range(3):
            router_gemm(x, w, out_bf16=False, split=8)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = router_gemm(x, w, out_bf16=False, split=8)
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x.float() @ w.T.contiguous().float(), atol=.00002, rtol=.00002)
    # Eleven target indexers plus one stored MTP indexer. Testing its weight
    # does not enable or qualify the MTP serving path.
    assert count == 12, count
