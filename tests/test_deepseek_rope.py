"""Inverse-RoPE completion regression for the portable gfx1201 decode path."""
import os
from types import SimpleNamespace

import pytest

from r9700_vllm.compat.deepseek_rope import wrap_decode


def test_completion_is_not_reported_by_portable_decode():
    calls = []
    def original(*args):
        calls.append(args)
        return 4
    wrapped = wrap_decode(original)
    args = tuple(object() for _ in range(8))
    assert wrapped(*args) == 0
    assert calls == [args + (None,)]
    with pytest.raises(RuntimeError, match="BF16"):
        wrapped(*args, output_mxfp8=object())
    assert len(calls) == 1


@pytest.mark.skipif(os.environ.get("R9700_DEEPSEEK_GPU_TEST") != "1",
                    reason="Explicit GPU test opt-in required")
@pytest.mark.parametrize("rows", [1, 4, 16])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("gpu", range(8))
def test_real_decode_and_mixed_batch_rotation(rows, compressed, gpu):
    import torch
    from test_deepseek_gpu import packed_cache, attention_reference
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention as Attention
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops

    if gpu >= torch.cuda.device_count():
        pytest.skip("GPU not present")
    torch.cuda.set_device(gpu)
    assert not ops._ON_GFX942 and not ops._ON_GFX950
    cache, kv = packed_cache(71)
    extra, extra_kv = packed_cache(72)
    g = torch.Generator().manual_seed(73)
    q = torch.randn(rows, 16, 512, generator=g).bfloat16()
    lens = torch.tensor([17 + i % 4 * 19 for i in range(rows)], dtype=torch.int32)
    idx = torch.full((rows, 128), -1, dtype=torch.int32)
    for i, n in enumerate(lens):
        idx[i, :n] = torch.randperm(256, generator=g)[:n]
    ragged = idx[idx >= 0].cuda()
    indptr = torch.cat([torch.zeros(1, dtype=torch.int32), lens.cumsum(0).int()]).cuda()
    positions = torch.arange(31, 31 + rows + 2, device="cuda", dtype=torch.int64)
    angles = torch.arange(128).float()[:, None] * torch.linspace(.015, .17, 32)[None, :]
    cos_sin = torch.cat([angles.cos(), angles.sin()], -1).cuda()
    sink = torch.linspace(-1, 1, 16)
    owner = SimpleNamespace(
        _use_aiter_sparse_mla=False, swa_cache_layer=SimpleNamespace(kv_cache=cache),
        _decode_topk_ragged=lambda **kwargs: (ragged, indptr, lens.cuda()),
        kv_cache_dtype="fp8_ds_mla", _has_kv_transfer=False,
        attn_sink=sink.cuda(), scale=512 ** -.5, head_dim=512,
        nope_head_dim=448, rope_head_dim=64,
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin))
    meta = SimpleNamespace(num_decodes=rows, num_decode_tokens=rows,
        decode_swa_indices=idx.cuda(), decode_swa_lens=lens.cuda(),
        decode_swa_ragged_indices=ragged, decode_swa_ragged_indptr=indptr)
    if compressed:
        reference_idx = torch.full((rows, 256), -1, dtype=torch.int32)
        for i, n in enumerate(lens):
            reference_idx[i, :n] = idx[i, :n]
            reference_idx[i, n:2*n] = idx[i, :n] + 256
        expected = attention_reference(q, torch.cat([kv, extra_kv]), reference_idx, lens*2, sink)
    else:
        expected = attention_reference(q, kv, idx, lens, sink)
    # Two independent prefill rows share the caller's final standalone pass.
    prefill = torch.randn(2, 16, 512, generator=g).bfloat16()
    expected = torch.cat([expected, prefill]).float()
    tail = expected[..., 448:].reshape(rows+2, 16, 32, 2).clone()
    cs = cos_sin.cpu()[positions.cpu()]
    a, b = tail[..., 0], tail[..., 1]
    expected[..., 448::2] = a * cs[:, None, :32] + b * cs[:, None, 32:]
    expected[..., 449::2] = b * cs[:, None, :32] - a * cs[:, None, 32:]
    expected = expected.bfloat16()
    original = Attention._forward_decode
    original = getattr(original, "__wrapped__", original)
    errors = []
    for method in (original, wrap_decode(original)):
        out = torch.empty(rows+2, 16, 512, dtype=torch.bfloat16, device="cuda")
        out[rows:].copy_(prefill.cuda())
        rotated = method(owner, q.cuda(), positions[:rows], extra if compressed else None,
                         meta, object() if compressed else None, not compressed, out[:rows])
        ops.rocm_inverse_rope_rows_(out[rotated:], positions[rotated:], cos_sin, 64)
        result = out.cpu()
        errors.append(float((result[:rows].float()-expected[:rows].float()).norm()
                            / expected[:rows].float().norm()))
        if method is not original:
            torch.testing.assert_close(result, expected, atol=.015, rtol=.02)
    assert errors[0] > .05, "The unpatched portable-path bug must reproduce"
    assert errors[1] < .005
