"""Opt-in paged-prefill numerical/allocation checks, no checkpoint loading.

Compare the real model prefill wrapper to independently decoded packed cache
and the original gather path. Exercise unequal C1/C2/C4, mixed decode/prefill
offsets, SWA-only, compression 1/2 and 32K history on every GPU.
"""
import os
from types import SimpleNamespace as NS, MethodType
from unittest.mock import patch

import pytest

if os.environ.get("R9700_DEEPSEEK_GPU_TEST") != "1":
    pytest.skip("Explicit GPU opt-in required", allow_module_level=True)

import torch
from vllm.models.deepseek_v41.amd import rocm
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
from r9700_vllm.compat.deepseek_prefill import wrap_prefill, wrap_workspace


def packed_cache(rows, block_size, seed):
    """CPU encoder independent of the upstream packed-cache writer/reader."""
    g = torch.Generator().manual_seed(seed)
    nb = (rows + block_size - 1) // block_size
    rows = nb * block_size
    values = torch.randn(rows, 512, generator=g).bfloat16()
    scales = torch.tensor([.25, .5, 1, 2, 4, .5, 1])[None, :, None]
    codes = (values[:, :448].float().reshape(rows, 7, 64) / scales).to(torch.float8_e4m3fn)
    decoded = values.clone()
    decoded[:, :448] = (codes.float() * scales).reshape(rows, 448)
    cache = torch.zeros(nb, block_size * 584, dtype=torch.uint8)
    data = cache[:, :block_size * 576].view(nb, block_size, 576)
    data[:, :, :448] = codes.view(torch.uint8).reshape(nb, block_size, 448)
    data[:, :, 448:] = values[:, 448:].contiguous().view(torch.uint8).reshape(nb, block_size, 128)
    exps = cache[:, block_size * 576:].view(nb, block_size, 8)
    exps[:, :, :7] = (scales.flatten().log2() + 127).byte()
    return cache.view(nb, block_size, 584).cuda(), decoded


def fixture(c, ratio, rows, context):
    g = torch.Generator().manual_seed(4132)
    # Unequal lengths; leading decode row ensures correct prefill token offsets.
    qlens = [rows // c + (i < rows % c) for i in range(c)]
    if c > 1 and qlens[-1] > 1:
        qlens[0] += 1; qlens[-1] -= 1
    full = [context - i * 137 for i in range(c)]
    block = 128
    # Keep independent sequence page tables, with shuffled physical pages.
    pages = (context + block - 1) // block
    bt = torch.randperm(c * pages, generator=g).reshape(c, pages).int()
    main, main_values = packed_cache(c * pages * block, block, 41)
    extra = extra_values = None
    cb = block // (ratio or 1)
    if ratio:
        extra, extra_values = packed_cache(c * pages * cb, cb, 42)
    starts = [0, 1]
    for n in qlens:
        starts.append(starts[-1] + n)
    lengths = torch.tensor([1] + full, dtype=torch.int32)
    query_start = torch.tensor(starts, dtype=torch.int32)
    topk = torch.full((1 + rows, 512), -1, dtype=torch.int32)
    swa = torch.full((rows, 128), -1, dtype=torch.int32)
    swa_lens = torch.empty(rows, dtype=torch.int32)
    expected_keys = []
    reqs = [0]
    row = 0
    for req, (n, length) in enumerate(zip(qlens, full)):
        for j in range(n):
            pos = length - n + j
            local = torch.arange(max(0, pos + 1 - 128), pos + 1)
            slots = bt[req, local // block] * block + local % block
            swa[row, :len(slots)] = slots
            swa_lens[row] = len(slots)
            selected = main_values[slots.long()]
            if ratio:
                count = min(512, (pos + 1) // ratio)
                k = torch.randperm((pos + 1) // ratio, generator=g)[:count]
                topk[1 + row, :count] = k.int()
                physical = bt[req, k // cb] * cb + k % cb
                selected = torch.cat((selected, extra_values[physical.long()]))
            expected_keys.append(selected)
            reqs.append(req + 1)
            row += 1
    # Decode's page table is unused by prefill.
    table = torch.cat((bt[:1], bt)).cuda()
    gather_lens = torch.tensor([min(length, n + 127) for length, n in zip(full, qlens)], dtype=torch.int32)
    def plan(**kwargs):
        result = []
        for i in range(c):
            N = full[i] // ratio if ratio else 0
            result.append((i, i + 1, N, N + int(gather_lens[i])))
        return result
    meta = NS(num_prefill_tokens=rows, num_decode_tokens=1, num_decodes=1,
        query_start_loc=query_start.cuda(), query_start_loc_cpu=query_start,
        token_to_req_indices=torch.tensor(reqs, device="cuda", dtype=torch.int32),
        seq_lens=lengths.cuda(), is_valid_token=torch.ones(rows+1, device="cuda", dtype=torch.bool),
        prefill_swa_indices=swa.cuda(), prefill_swa_lens=swa_lens.cuda(),
        prefill_seq_lens=lengths[1:].cuda(), prefill_seq_lens_cpu=lengths[1:],
        prefill_gather_lens=gather_lens.cuda(),
        block_table=table, block_size=block, get_prefill_chunk_plan=plan)
    meta.prefill_swa_ragged_indices, meta.prefill_swa_ragged_indptr = ops.build_ragged_indices_from_dense(
        meta.prefill_swa_indices, meta.prefill_swa_lens)
    attn = NS(block_table=table, block_size=block) if ratio else None
    cls = rocm.DeepseekV41ROCMAiterMLAAttention
    owner = NS(_r9700_paged_prefill=True, _use_aiter_sparse_mla=False,
        kv_cache_dtype="fp8_ds_mla", head_dim=512, nope_head_dim=448, rope_head_dim=64,
        compress_ratio=ratio, window_size=128, PREFILL_CHUNK_SIZE=1,
        max_model_len=context, max_num_batched_tokens=rows, n_local_heads=8,
        topk_indices_buffer=topk.cuda(), _index_source_prefix="source",
        compressed_cache_prefix="compressed", scale=512 ** -.5,
        attn_sink=torch.linspace(-1, 1, 8, device="cuda"))
    owner._static_forward_context = {"source": owner}
    owner._prefill_topk_ragged = MethodType(cls._prefill_topk_ragged, owner)
    owner._prefill_workspace_shapes = MethodType(cls._prefill_workspace_shapes, owner)
    q = torch.randn(rows, 8, 512, generator=g).bfloat16().cuda()
    return owner, q, extra, main, attn, meta, expected_keys


@pytest.mark.parametrize("gpu", range(8))
@pytest.mark.parametrize("c,ratio,rows,context", [
    (1, 0, 8, 1024), (2, 1, 16, 2048), (4, 2, 32, 4096),
    (1, 2, 512, 32768), (4, 2, 512, 32768), (4, 1, 512, 32768),
])
def test_paged_matches_reference_and_gather(gpu, c, ratio, rows, context):
    torch.cuda.set_device(gpu)
    assert torch.cuda.get_device_properties(gpu).gcnArchName.split(":")[0] == "gfx1201"
    owner, q, extra, main, attn, meta, keys = fixture(c, ratio, rows, context)
    original = rocm.DeepseekV41ROCMAiterMLAAttention._forward_prefill
    paged = wrap_prefill(original, ops._rocm_sparse_attn_decode_ragged_triton)
    output = torch.empty_like(q)
    positions = torch.arange(rows, device="cuda")
    args = (owner, q, positions, extra, main, output, attn, meta)
    paged(*args)
    torch.cuda.synchronize()
    # Check every row against independent FP32 attention on CPU-decoded bytes.
    expected = []
    for start in range(0, rows, 16):
        k = torch.stack(keys[start:start+16]).cuda().float()
        scores = torch.bmm(q[start:start+16].float(), k.transpose(1, 2)) * owner.scale
        sink = owner.attn_sink[None, :, None].expand(len(k), -1, 1)
        probs = torch.cat((scores, sink), -1).softmax(-1)[:, :, :-1]
        expected.append(torch.bmm(probs, k).bfloat16())
    torch.testing.assert_close(output, torch.cat(expected), atol=.008, rtol=.02)
    # Compare to the exact former model prefill with full BF16 gathering.
    class Workspace:
        def get_simultaneous(self, *shapes):
            return [torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in shapes]
    baseline = torch.empty_like(q)
    with patch.object(rocm, "current_workspace_manager", return_value=Workspace()):
        original(owner, q, positions, extra, main, baseline, attn, meta)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, baseline, atol=.008, rtol=.02)
    # After warmup, paged scratch must be bounded by the query/top-k sizes.
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    paged(*args)
    torch.cuda.synchronize()
    scratch = torch.cuda.max_memory_allocated() - before
    assert scratch < 16 * 1024**2, scratch
    shapes = wrap_workspace(rocm.DeepseekV41ROCMAiterMLAAttention._prefill_workspace_shapes)
    assert shapes(owner, 524288, rows, q) == []
    print(f"gpu={gpu} C{c} ratio={ratio} rows={rows} context={context} scratch={scratch}")
