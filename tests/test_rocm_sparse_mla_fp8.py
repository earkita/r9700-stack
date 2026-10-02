# SPDX-License-Identifier: Apache-2.0
"""Upstream-only FP8 sparse MLA gates; no r9700 plugin or checkpoint imports.

Run against the patched vLLM image on RDNA4. Numerical tolerances compare the
reader to the same quantized cache, not lossless equivalence to BF16 weights.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")


@pytest.mark.parametrize("dtype", ["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_ds_mla"])
@pytest.mark.parametrize("rdna", [False, True])
@pytest.mark.parametrize("rows", [1, 8, 16])
def test_dispatch(dtype, rdna, rows):
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import _use_rocm_sparse_triton
    args = dict(kv_cache_dtype=dtype, head_size=512, kv_lora_rank=512,
                num_prefills=int(rows > 8), num_decodes=int(rows <= 8),
                num_decode_tokens=rows if rows <= 8 else 0, max_query_len=rows,
                fp8_triton_supported=rdna)
    expected = dtype in ("auto", "bfloat16") or (rdna and dtype in ("fp8", "fp8_e4m3"))
    assert _use_rocm_sparse_triton(**args) == expected
    assert not _use_rocm_sparse_triton(**(args | dict(head_size=576)))
    assert not _use_rocm_sparse_triton(**(args | dict(num_prefills=0, num_decodes=0)))


def inputs(rows, scale, sink):
    torch.manual_seed(1701)
    q = torch.randn(rows, 8, 512, device="cuda", dtype=torch.bfloat16)
    kv_scale = torch.tensor(scale, device="cuda", dtype=torch.float32)
    source = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    kv = (source.float() / kv_scale).to(current_platform.fp8_dtype())[:, None, :]
    indices = torch.stack([torch.randperm(256, device="cuda")[:67] for _ in range(rows)]).int()
    indices[:, -3:] = -1
    if rows > 1:
        indices[-1] = -1  # Empty row, also with a sink.
    lengths = (indices >= 0).sum(-1, dtype=torch.int32)
    sinks = torch.randn(8, device="cuda") if sink else None
    return q, kv, kv_scale, indices, lengths, sinks


def reference(q, kv, kv_scale, indices, sinks):
    dequant = (kv.float() * kv_scale).to(torch.bfloat16)[:, 0].float()
    rows = []
    for row, idx in enumerate(indices):
        selected = dequant[idx[idx >= 0].long()]
        scores = q[row].float() @ selected.T * 512 ** -.5
        if sinks is not None:
            probs = torch.cat([scores, sinks[:, None]], dim=1).softmax(-1)[:, :-1]
        else:
            probs = scores.softmax(-1)
        rows.append(probs @ selected)
    return torch.stack(rows).to(q.dtype)


def call(q, kv, kv_scale, indices, lengths, sinks, ragged=False):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill
    extra = {}
    if ragged:
        extra = dict(ragged_indices=indices[indices >= 0],
                     ragged_indptr=torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32),
                                               lengths.cumsum(0).int()]))
    out = torch.empty_like(q)
    rocm_sparse_attn_prefill(q=q, kv=kv, indices=indices, topk_length=lengths,
                            scale=512 ** -.5, head_dim=512, nope_head_dim=512,
                            rope_head_dim=0, attn_sink=sinks, output=out,
                            kv_scale=kv_scale, **extra)
    return out


@pytest.mark.parametrize("rows", [1, 8, 16])
@pytest.mark.parametrize("scale", [.125, 1., 2.5])
@pytest.mark.parametrize("sink,ragged", [(False, False), (True, True)])
def test_reader(rows, scale, sink, ragged):
    args = inputs(rows, scale, sink)
    q_before = args[0].clone()
    actual = call(*args, ragged=ragged)
    expected = reference(*args[:4], args[-1])
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=3e-3)
    torch.testing.assert_close(args[0], q_before, rtol=0, atol=0)
    # Reading FP8 must agree with the existing BF16 reader after dequantization.
    dequant = (args[1].float() * args[2]).to(torch.bfloat16)
    bf16 = call(args[0], dequant, None, *args[3:], ragged=ragged)
    torch.testing.assert_close(actual, bf16, rtol=0, atol=0)


def test_graph_replay_reads_changed_scale_and_query():
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _rocm_sparse_attn_prefill_ragged_triton
    q, kv, scale, indices, lengths, sinks = inputs(8, .125, True)
    # Runtime metadata is preallocated. Boolean indexing here would synchronize
    # to determine an allocation size and is not legal during graph capture.
    flat = indices[indices >= 0]
    indptr = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32), lengths.cumsum(0).int()])
    def invoke():
        return _rocm_sparse_attn_prefill_ragged_triton(
            q, kv[:, 0], flat, indptr, 512 ** -.5, sinks, 512, 0, scale)
    for _ in range(3):
        invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = invoke()
    q.normal_()
    scale.fill_(.25)
    kv.copy_((torch.randn_like(kv.float()) * 3).to(kv.dtype))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference(q, kv, scale, indices, sinks), rtol=1e-2, atol=3e-3)


@pytest.mark.parametrize("bad", ["missing", "dtype", "shape", "cpu", "query", "e5m2"])
def test_invalid_inputs(bad):
    q, kv, scale, indices, lengths, sinks = inputs(1, 1., False)
    if bad == "missing": scale = None
    if bad == "dtype": scale = scale.to(torch.bfloat16)
    if bad == "shape": scale = scale.repeat(2)
    if bad == "cpu": scale = scale.cpu()
    if bad == "query": q = q.to(kv.dtype)
    if bad == "e5m2": kv = kv.float().to(torch.float8_e5m2)
    with pytest.raises(ValueError):
        call(q, kv, scale, indices, lengths, sinks)


@pytest.mark.parametrize("scale_value", [.125, 2.5])
def test_upstream_cache_writer_roundtrip(scale_value):
    from vllm import _custom_ops as ops
    q, _, scale, indices, lengths, sinks = inputs(8, scale_value, False)
    source = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    # Non-contiguous physical page placement and unchanged unused slots.
    slots = torch.randperm(512, device="cuda")[:256].long()
    cache = torch.zeros(8, 64, 512, device="cuda", dtype=torch.uint8)
    ops.concat_and_cache_mla(source, source[:, :0], cache, slots, "fp8_e4m3", scale)
    decoded = cache.view(current_platform.fp8_dtype()).view(-1, 1, 512)
    expected_bytes = (source.float() / scale).to(current_platform.fp8_dtype()).view(torch.uint8)
    torch.testing.assert_close(decoded.view(torch.uint8)[slots, 0], expected_bytes, rtol=0, atol=0)
    physical = torch.where(indices >= 0, slots[indices.clamp_min(0).long()], -1).int()
    actual = call(q, decoded, scale, physical, lengths, sinks, ragged=True)
    torch.testing.assert_close(actual, reference(q, decoded, scale, physical, sinks), rtol=1e-2, atol=3e-3)


@pytest.mark.parametrize("tuple_query", [False, True])
def test_backend_keeps_query_bf16(monkeypatch, tuple_query):
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as backend
    monkeypatch.setattr(backend, "_use_rocm_sparse_triton", lambda **kwargs: True)
    monkeypatch.setattr(backend, "fit_kpool_indices_to_aiter", lambda *args: args[0])
    monkeypatch.setattr(backend, "triton_convert_req_index_to_global_index", Mock())
    monkeypatch.setattr(backend.ops, "scaled_fp8_quant", Mock(side_effect=AssertionError("Q quantized")))
    impl = backend.ROCMAiterMLASparseImpl.__new__(backend.ROCMAiterMLASparseImpl)
    impl.kv_cache_dtype, impl.kv_lora_rank, impl.num_heads = "fp8_e4m3", 512, 8
    q, kv, _, indices, _, _ = inputs(8, 1., False)
    impl.q_concat_buffer = torch.empty_like(q)
    impl.topk_indices_buffer = indices
    impl._forward_mla = Mock(return_value=(q, None))
    meta = SimpleNamespace(num_prefills=0, num_decodes=1, num_decode_tokens=8,
                           max_query_len=8, num_actual_tokens=8, topk_tokens=67,
                           req_id_per_token=None, block_table=None, paged_kv_indptr=None,
                           paged_kv_indices=None, block_size=64)
    layer = SimpleNamespace(_decode_concat_quant_fp8_op=Mock(side_effect=AssertionError("Q quantized")))
    impl.forward_mqa((q, q[:, :, :0]) if tuple_query else q, kv.view(torch.uint8), meta, layer)
    forwarded = impl._forward_mla.call_args.args
    assert forwarded[1].dtype == torch.bfloat16
    assert forwarded[2].dtype == current_platform.fp8_dtype()
    assert forwarded[-1] is True
