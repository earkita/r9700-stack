"""Numerical gate for the upstream RDNA4 indexer path used by the GLM adapter.

Run in the ROCm image, VLLM_ROCM_USE_AITER=1. This is a logits/reference gate,
not a claim about full kpool selection, attention, KDA, or model-level quality.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")


@pytest.mark.parametrize("rows,keys", [(1, 512), (16, 8192)])
def test_upstream_indexer_logits_against_torch(rows, keys):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_fp8_mqa_logits
    assert rocm_aiter_ops.is_rdna_aiter_enabled(), "Enable VLLM_ROCM_USE_AITER=1 on gfx1201"
    torch.manual_seed(41)
    q = torch.randn(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(keys, device="cuda", dtype=torch.float32) + 0.01
    weights = torch.randn(rows, 32, device="cuda", dtype=torch.float32) / 32
    starts = torch.arange(rows, device="cuda", dtype=torch.int32)
    ends = torch.full((rows,), keys - 3, device="cuda", dtype=torch.int32)
    actual = rocm_fp8_mqa_logits(q, (k, scales), weights, starts, ends)
    # FP8 values are exact inputs; compare FP32 accumulation and weighted ReLU.
    # Upstream fp8_mqa_logits_torch casts to BF16 and rounds the einsum OUTPUT
    # to BF16 before scaling. That is a different rounding point than Triton.
    scores = torch.einsum("mhd,nd->hmn", q.float(), k.float()) * scales
    reference = (scores.relu() * weights.T[:, :, None]).sum(dim=0)
    columns = torch.arange(keys, device="cuda")
    valid = (columns[None, :] >= starts[:, None]) & (columns[None, :] < ends[:, None])
    assert actual.shape == reference.shape
    assert torch.isfinite(actual[valid]).all()
    torch.testing.assert_close(actual[valid], reference[valid], rtol=5e-3, atol=1e-3)
    assert torch.isneginf(actual[~valid]).all()


@pytest.mark.parametrize("rows", [1, 16])
def test_upstream_sparse_mla_output_against_torch(rows):
    """GLM absorbed MLA dimensions: 8 TP8 heads, 512 latent, no RoPE.

    Tests sparse attention on already selected/gathered keys, including padded
    indices; separate cache-geometry gates cover physical address mapping.
    """
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill
    torch.manual_seed(53)
    q = torch.randn(rows, 8, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(4096, 1, 512, device="cuda", dtype=torch.bfloat16)
    indices = torch.stack([torch.randperm(4096, device="cuda")[:2048] for _ in range(rows)]).int()
    indices[:, -7:] = -1
    lengths = (indices >= 0).sum(-1, dtype=torch.int32)
    out = torch.empty_like(q)
    scale = 512 ** -.5
    rocm_sparse_attn_prefill(q=q, kv=kv, indices=indices, topk_length=lengths,
                            scale=scale, head_dim=512, nope_head_dim=512,
                            rope_head_dim=0, attn_sink=None, output=out)
    expected = []
    for row in range(rows):
        selected = kv[indices[row, :lengths[row]].long(), 0].float()
        scores = (q[row].float() @ selected.T) * scale
        expected.append(scores.softmax(-1) @ selected)
    reference = torch.stack(expected).to(out.dtype)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-3)
