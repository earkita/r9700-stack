# SPDX-License-Identifier: Apache-2.0
"""Plugin FP8 gates on unmodified pinned vLLM, without model weights."""
from types import SimpleNamespace
import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('vllm')
from vllm.platforms import current_platform
pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason='ROCm required')

def inputs(rows, scale, sink, heads=8):
    torch.manual_seed(1701)
    q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16)
    kv_scale = torch.tensor(scale, device="cuda", dtype=torch.float32)
    source = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    kv = (source.float() / kv_scale).to(current_platform.fp8_dtype())[:, None, :]
    indices = torch.stack([torch.randperm(256, device="cuda")[:67] for _ in range(rows)]).int()
    indices[:, -3:] = -1
    if rows > 1:
        indices[-1] = -1  # Empty row, also with a sink.
    lengths = (indices >= 0).sum(-1, dtype=torch.int32)
    sinks = torch.randn(heads, device="cuda") if sink else None
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
    from r9700_vllm.attn.glm_sparse_fp8 import rocm_sparse_attn_prefill
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
@pytest.mark.parametrize("heads", [8, 16])
def test_reader(rows, scale, sink, ragged, heads):
    args = inputs(rows, scale, sink, heads=heads)
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
    from r9700_vllm.attn.glm_sparse_fp8 import _rocm_sparse_attn_prefill_ragged_triton
    q, kv, scale, indices, lengths, sinks = inputs(8, .125, True, heads=16)
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

@pytest.mark.parametrize("rows,heads,fp8,rdna,expected", [
    (1, 16, True, True, 8), (8, 16, True, True, 8),
    (9, 16, True, True, 4), (8, 32, True, True, 4),
    (8, 8, True, True, 4), (8, 16, False, True, 4),
    (8, 16, True, False, 4),
])
def test_launch_tuning_is_scoped(monkeypatch, rows, heads, fp8, rdna, expected):
    from r9700_vllm.attn import glm_sparse_fp8 as ops
    args = list(inputs(rows, 1., False, heads=heads))
    if not fp8:
        args[1] = args[1].to(torch.bfloat16)
        args[2] = None
    launches = []
    class RecordLaunch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(kwargs)
            return launch
    monkeypatch.setattr(ops, "_ON_RDNA4", rdna)
    monkeypatch.setattr(ops, "_sparse_attn_prefill_ragged_kernel", RecordLaunch())
    call(*args, ragged=True)
    assert len(launches) == 1
    assert launches[0]["num_warps"] == expected
    assert launches[0]["BLOCK_K"] == 16

@pytest.mark.parametrize("scale_value", [.125, 1., 2.5])
def test_decode_long_sparse_rows_match_bf16_reference(scale_value):
    torch.manual_seed(707)
    q = torch.randn(8, 16, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(32768, 1, 512, device="cuda").to(current_platform.fp8_dtype())
    scale = torch.tensor(scale_value, device="cuda", dtype=torch.float32)
    indices = torch.stack([torch.randperm(32768, device="cuda")[:2048] for _ in range(8)]).int()
    indices[0, -5:] = -1
    indices[-1] = -1
    lengths = (indices >= 0).sum(-1, dtype=torch.int32)
    sinks = torch.randn(16, device="cuda")
    actual = call(q, kv, scale, indices, lengths, sinks, ragged=True)
    expected = reference(q, kv, scale, indices, sinks)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=3e-3)
    bf16 = call(q, (kv.float() * scale).to(torch.bfloat16), None,
                indices, lengths, sinks, ragged=True)
    torch.testing.assert_close(actual, bf16, rtol=0, atol=0)


@pytest.mark.parametrize('family,dtype,heads,size,rank,expected', [
    ('glm5_next','fp8_e4m3',8,512,512,True),
    ('glm5_next','fp8',8,512,512,True),
    ('glm5_next_text','fp8_e4m3',8,512,512,True),
    ('qwen','fp8_e4m3',8,512,512,False),
    ('glm5_next','bfloat16',8,512,512,False),
    ('glm5_next','fp8_e5m2',8,512,512,False),
    ('glm5_next','fp8_ds_mla',8,512,512,False),
    ('glm5_next','fp8_e4m3',16,512,512,False),
    ('glm5_next','fp8_e4m3',8,576,512,False),
])
def test_instance_scope(family,dtype,heads,size,rank,expected):
    from r9700_vllm.compat.glm_sparse_fp8 import eligible
    cfg=SimpleNamespace(hf_config=SimpleNamespace(model_type=family))
    assert eligible(cfg,dtype,heads,size,rank) is expected


# The cloned builder resolves this binding from its own private globals.
def _use_rocm_sparse_triton(**kwargs):
    return 'original'


@pytest.mark.parametrize('family',['glm5_next','glm5_next_text'])
def test_builder_predicate_is_private(family):
    from r9700_vllm.compat.glm_sparse_fp8 import _private_build
    def build(self):
        return _use_rocm_sparse_triton(kv_cache_dtype=self.kv_cache_dtype,
            head_size=512,kv_lora_rank=512,num_prefills=0,num_decodes=1)
    wrapped=_private_build(build)
    hf=SimpleNamespace(model_type=family)
    obj=SimpleNamespace(model_config=SimpleNamespace(hf_config=hf),
        kv_cache_dtype='fp8_e4m3',num_heads=8,
        mla_dims=SimpleNamespace(kv_lora_rank=512,qk_rope_head_dim=0))
    assert wrapped(obj) is True
    assert build(obj)=='original'
    hf.model_type='qwen'
    assert wrapped(obj)=='original'
    hf.model_type=family;obj.kv_cache_dtype='bfloat16'
    assert wrapped(obj)=='original'


def test_patch_scoping_idempotence_and_file_integrity(monkeypatch):
    import hashlib
    from pathlib import Path
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as b
    from r9700_vllm.compat import glm_sparse_fp8 as adapter
    impl=b.ROCMAiterMLASparseImpl;builder=b.ROCMAiterMLASparseMetadataBuilder
    # Ensure the fixture restores these after installation.
    for obj,name in [(impl,'__init__'),(impl,'forward_mqa'),(builder,'build')]:
        monkeypatch.setattr(obj,name,getattr(obj,name))
    predicate=b._use_rocm_sparse_triton
    before=hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest()
    monkeypatch.setenv('R9K_GLM_FP8_SPARSE','0')
    assert adapter.patch() is False
    monkeypatch.setenv('R9K_GLM_FP8_SPARSE','1')
    assert adapter.patch() is True
    installed=impl.forward_mqa
    assert adapter.patch() is True and installed is impl.forward_mqa
    assert b._use_rocm_sparse_triton is predicate
    assert hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest()==before


def test_wrong_pin_rejected(monkeypatch):
    from r9700_vllm.compat import gate,glm_sparse_fp8 as adapter
    monkeypatch.setenv('R9K_GLM_FP8_SPARSE','1')
    monkeypatch.setattr(gate,'vllm_commit',lambda:'other-version')
    with pytest.raises(RuntimeError,match='audited vLLM'):
        adapter.patch()


@pytest.mark.parametrize('family,dtype,expected',[
    ('glm5_next_text','fp8_e4m3','local'),
    ('glm5_next','fp8_e4m3','local'),
    ('glm5_next_text','bfloat16','upstream'),
    ('qwen','fp8_e4m3','upstream'),
])
def test_constructor_dispatch_under_nested_model_config(monkeypatch,family,dtype,expected):
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as b
    from r9700_vllm.compat import glm_sparse_fp8 as adapter
    impl=b.ROCMAiterMLASparseImpl
    def init(self):
        self.num_heads=8
        self.head_size=self.kv_lora_rank=512
        self.kv_cache_dtype=dtype
    monkeypatch.setattr(impl,'__init__',init)
    monkeypatch.setattr(impl,'forward_mqa',lambda *args:'upstream')
    monkeypatch.setattr(b.ROCMAiterMLASparseMetadataBuilder,'build',
                        b.ROCMAiterMLASparseMetadataBuilder.build)
    cfg=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=family)))
    monkeypatch.setattr(b,'get_current_vllm_config',lambda:cfg)
    monkeypatch.setattr(adapter,'forward_fp8',lambda *args:'local')
    monkeypatch.setenv('R9K_GLM_FP8_SPARSE','1')
    assert adapter.patch()
    obj=impl()
    assert obj.forward_mqa(None,None,None,None)==expected


@pytest.mark.parametrize('tuple_query',[False,True])
def test_forward_keeps_bf16_query_and_scale(monkeypatch,tuple_query):
    from unittest.mock import Mock
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as b
    from r9700_vllm.compat.glm_sparse_fp8 import forward_fp8
    q,kv,scale,indices,lengths,sinks=inputs(8,.125,True)
    monkeypatch.setattr(b,'fit_kpool_indices_to_aiter',lambda x,n:x)
    monkeypatch.setattr(b,'triton_convert_req_index_to_global_index',Mock())
    monkeypatch.setattr(b.ops,'scaled_fp8_quant',Mock(side_effect=AssertionError('Q quantized')))
    obj=SimpleNamespace(num_heads=8,kv_lora_rank=512,scale=512**-.5,
        q_concat_buffer=torch.empty_like(q),topk_indices_buffer=indices,sinks=sinks)
    meta=SimpleNamespace(num_actual_tokens=8,topk_tokens=67,block_size=64,
        req_id_per_token=None,block_table=None,attn_out_dtype=torch.bfloat16,
        paged_kv_indices=indices[indices>=0],
        paged_kv_indptr=torch.cat([torch.zeros(1,device='cuda',dtype=torch.int32),lengths.cumsum(0).int()]))
    layer=SimpleNamespace(_k_scale=scale)
    actual,lse=forward_fp8(obj,(q,q[:,:,:0]) if tuple_query else q,kv.view(torch.uint8),meta,layer)
    assert lse is None
    torch.testing.assert_close(actual,reference(q,kv,scale,indices,sinks),rtol=1e-2,atol=3e-3)
