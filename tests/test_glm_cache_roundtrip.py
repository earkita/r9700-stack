"""GPU writer -> virtual pages -> prefill gather / decode logits regression.

Exercise the actual pinned GLM helpers with non-contiguous physical pages and
chunk boundaries. Compression is shared with the reference; addressing and
decode arithmetic are checked independently of the cache reader.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")


@pytest.fixture(params=[640, 1280])
def written_cache(request):
    from vllm.models.glm5next.amd.sparse_indexer import _kpool_compress_insert
    from vllm.models.glm5next.amd.ops.kpool_compress import kpool_compress_and_write_cache
    from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
    from vllm.v1.worker.utils import select_common_block_size
    from r9700_vllm.attn.glm_indexer import GlmKpool4IndexerBackend

    torch.manual_seed(912)
    manager = request.param
    selected = select_common_block_size(manager, [GlmKpool4IndexerBackend])
    page, factor = selected // 4, manager // selected
    chunks = [1920, 1920, 1920, 1920, 1280, 472]
    n, dim = sum(chunks), 128
    pools = n // 4
    ids = torch.randperm(63, device="cuda", dtype=torch.int32) + 1
    table = (ids[:, None] * factor + torch.arange(factor, device="cuda")).reshape(1, -1).int()
    cache = torch.zeros((64 * factor, page, dim + 4), device="cuda", dtype=torch.uint8)
    k = torch.randn((n, dim), device="cuda", dtype=torch.bfloat16)
    gates = torch.randn_like(k)
    ape = torch.randn((4, dim), device="cuda")
    ref_k, ref_s = kpool_compress_and_write_cache(
        cache, k.view(pools, 4, dim), gates.view(pools, 4, dim), ape,
        torch.zeros(pools, device="cuda", dtype=torch.int64), 4,
        return_compressed=True, write_cache=False)
    start = 0
    for chunk in chunks:
        end = start + chunk
        pos = torch.arange(start, end, device="cuda")
        slots = ids[(pos // manager).long()].long() * manager + pos % manager
        compressed = get_compressed_slot_mapping(
            chunk, slots, torch.tensor([0, chunk], device="cuda", dtype=torch.int32),
            torch.tensor([end], device="cuda", dtype=torch.int32), table, page, 4)
        expected = ids[(pos // manager).long()].long() * (manager // 4) + pos % manager // 4
        expected = torch.where(pos % 4 == 3, expected, -1)
        torch.testing.assert_close(compressed, expected, rtol=0, atol=0)
        _kpool_compress_insert(k[start:end], gates[start:end], ape, cache, compressed, 4, dim, True)
        start = end
    return cache, table, ref_k, ref_s


def test_prefill_gather_roundtrip(written_cache):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import cp_gather_indexer_k_quant_cache_triton
    cache, table, ref_k, ref_s = written_cache
    n = ref_k.shape[0]
    actual_k, actual_s = torch.empty_like(ref_k), torch.empty_like(ref_s)
    cp_gather_indexer_k_quant_cache_triton(
        cache, actual_k, actual_s, table,
        torch.tensor([0, n], device="cuda", dtype=torch.int32),
        torch.zeros(n, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(actual_k.view(torch.uint8), ref_k.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(actual_s, ref_s, rtol=0, atol=0)


def test_decode_logits_roundtrip(written_cache):
    from vllm.v1.worker.workspace import init_workspace_manager
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse
    from r9700_vllm.compat.glm_paged_logits import patch
    assert patch()
    init_workspace_manager(torch.cuda.current_device())
    cache, table, ref_k, ref_s = written_cache
    n, dim = ref_k.shape
    q = (torch.randn(1, 1, 32, dim, device="cuda") * 2).to(ref_k.dtype)
    weights = torch.rand(1, 32, device="cuda")
    context = torch.tensor([n], device="cuda", dtype=torch.int32)
    actual = rocm_aiter_mla_sparse.rocm_fp8_paged_mqa_logits(
        q, cache.unsqueeze(2), weights, context, table,
        torch.empty(0, device="cuda", dtype=torch.int32), max_model_len=4096)
    dot = q[0, 0].float() @ ref_k.float().T
    ref = ((dot * ref_s.reshape(1, n)).relu() * weights[0, :, None]).sum(0)
    torch.testing.assert_close(actual[0, :n], ref, rtol=3e-5, atol=.003)
    assert torch.isneginf(actual[0, n:]).all()
    assert set(actual[0, :n].topk(512).indices.tolist()) == set(ref.topk(512).indices.tolist())


@pytest.mark.parametrize("per_row", [False, True])
def test_decode_graph_replay_and_context_mask(written_cache, per_row):
    from vllm.v1.worker.workspace import init_workspace_manager
    from r9700_vllm.attn.glm_paged_logits import paged_logits
    init_workspace_manager(torch.cuda.current_device())
    cache, table, ref_k, ref_s = written_cache
    n = ref_k.shape[0]
    # Multiple sequences/steps; weights can be negative. Replay also exercises
    # a new empty context and a partial page without rebuilding the graph.
    q = torch.randn(2, 2, 32, 128, device="cuda").to(ref_k.dtype)
    weights = torch.randn(4, 32, device="cuda")
    lengths = torch.tensor([[n - 1, n], [64, 65]] if per_row else [n, 65],
                           device="cuda", dtype=torch.int32)
    tables = table.repeat(2, 1)
    call = lambda: paged_logits(q, cache.unsqueeze(2), weights, lengths, tables, None, 4093)
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    q.copy_((torch.randn_like(q.float()) * 3).to(q.dtype))
    lengths.copy_(torch.tensor([[n - 1, n], [0, 17]] if per_row else [n, 17], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    for row in range(4):
        limit = int(lengths.flatten()[row if per_row else row // 2])
        if not per_row:
            limit = limit - 2 + row % 2 + 1
        dot = q.reshape(4, 32, 128)[row].float() @ ref_k[:limit].float().T
        ref = ((dot * ref_s[:limit].reshape(1, -1)).relu() * weights[row, :, None]).sum(0)
        torch.testing.assert_close(out[row, :limit], ref, rtol=3e-5, atol=.003)
        assert torch.isneginf(out[row, limit:]).all()


def test_dispatch_preserves_non_glm_shapes_and_compression():
    from unittest.mock import Mock
    from r9700_vllm.compat.glm_paged_logits import _wrap
    original, fixed = Mock(), Mock()
    dispatch = _wrap(original, fixed)
    q = torch.empty((1, 1, 32, 128), dtype=torch.float8_e4m3fn)
    cache = torch.empty((1, 32, 1, 132), dtype=torch.uint8)
    dispatch(q, cache, None, None, None, None, 1024, compress_ratio=4)
    dispatch(q[:, :, :16], cache, None, None, None, None, 1024)
    assert original.call_count == 2
    fixed.assert_not_called()
