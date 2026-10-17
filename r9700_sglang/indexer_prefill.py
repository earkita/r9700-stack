# SPDX-License-Identifier: Apache-2.0
"""Opt-in bounded query rows for the pinned GLM KPool prefill plan.

The ordinary upstream Indexer guards its MQA logits allocation, but KPool's
plan entry bypasses it. AITER allocates FP32 logits with a 256-key row stride;
8192 queries at 256K context reach its 2-GiB addressing limit. Keep key gathering
and every original transform/mapping, processing independent query slices.
This can change AITER's M-dependent tile heuristic: GPU numerical/top-k and
full-model qualification are required. Decode/verify and the default are intact.
"""
import os


def row_slices(queries, keys, row_limit, byte_limit=2**31-1):
    if min(queries,keys,row_limit,byte_limit) <= 0:
        raise ValueError("Positive query/key counts, row and byte limits required")
    stride = (keys+255)//256*256
    limit = min(row_limit, byte_limit//(stride*4))
    if limit < 1:
        raise ValueError("One aligned logits row exceeds the declared byte limit")
    return [slice(start,min(start+limit,queries)) for start in range(0,queries,limit)]


def apply(indexer, layer_id, q_fp8, weights, metadata):
    """Return None for unchanged dispatch, otherwise the completed top-k tensor."""
    value = os.environ.get("R9K_SGLANG_INDEXER_PREFILL_ROWS", "0")
    if value == "0":
        return None
    if not value.isdecimal() or int(value) not in (512,1024,2048):
        raise ValueError("R9K_SGLANG_INDEXER_PREFILL_ROWS must be 0, 512, 1024 or 2048")
    import torch
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import gather_index_k_scale_prefix_into
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
    from sglang.srt.model_executor.runner import get_is_capture_mode

    if (get_is_capture_mode() or torch.version.hip is None
            or torch.cuda.get_device_properties(q_fp8.device).gcnArchName.split(":")[0] != "gfx1201"
            or indexer.index_kpool != 4 or indexer.index_topk != 2048
            or indexer.n_heads != 32 or indexer.head_dim != 128):
        raise RuntimeError("Bounded KPool prefill requires eager gfx1201 GLM KPool4/topk2048")
    plan = metadata.attn_metadata.kpool_extend_plan
    n_real = plan.seq_lens_expanded.shape[0]
    keys = plan.ragged_total_k_rows
    if n_real == 0 or keys == 0:
        return None
    if weights.ndim != 3 or weights.shape[-1] != 1 or n_real > q_fp8.shape[0]:
        raise ValueError("Unexpected KPool plan query/weight layout")
    chunks = row_slices(n_real,keys,int(value))
    if len(chunks) == 1:
        return None

    k_u8, k_scale = plan.ragged_k_u8, plan.ragged_k_scale
    if k_u8 is None or k_scale is None:
        raise ValueError("KPool plan has no gathered-key storage")
    pool = get_token_to_kv_pool()
    gather_index_k_scale_prefix_into(pool=pool,buf=indexer._get_index_k_read_buffer(pool,layer_id),
        page_indices=plan.ragged_concat_page_table,seq_len=keys,k_out=k_u8,scale_out=k_scale)
    k_fp8 = k_u8.view(torch.float8_e4m3fn).contiguous()
    k_scale = k_scale.contiguous()
    weights = weights.squeeze(-1)
    return select_rows(indexer,q_fp8,k_fp8,k_scale,weights,plan,metadata,chunks)


def select_rows(indexer,q_fp8,k_fp8,k_scale,weights,plan,metadata,chunks):
    """Original MQA/selection operators with bounded row storage and mappings.

    Kept separate from key gathering to qualify the actual row-sliced operators
    against full MQA/selection on synthetic unequal-length cache plans.
    """
    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod

    n_real = plan.seq_lens_expanded.shape[0]
    page_table = row_index = offsets = None
    if envs.SGLANG_DSA_FUSE_TOPK.get():
        if metadata.topk_transform_method == TopkTransformMethod.PAGED:
            page_table = plan.ragged_paged_page_table
            row_index = plan.ragged_paged_page_table_row_index
        elif metadata.topk_transform_method == TopkTransformMethod.RAGGED:
            offsets = metadata.attn_metadata.topk_indices_offset

    result = None
    for rows in chunks:
        logits = fp8_mqa_logits(q_fp8[rows].contiguous(),k_fp8,k_scale,weights[rows].contiguous(),
                               plan.ragged_q_ks[rows],plan.ragged_q_ke[rows],clean_logits=True)
        local_table = page_table
        if page_table is not None and row_index is None:
            if page_table.shape[0] < n_real:
                raise ValueError("A shared page table requires explicit query-to-table row indices")
            local_table = page_table[rows]
        selected = indexer._topk_from_kpool_logits(logits,plan.pooled_seq_lens_expanded[rows],
            seq_lens=plan.seq_lens_expanded[rows],page_table=local_table,
            page_table_row_index=row_index[rows] if row_index is not None else None,
            topk_offsets=offsets[rows] if offsets is not None else None,
            row_starts=plan.ragged_q_ks[rows],out_rows=rows.stop-rows.start)
        if selected.shape[0] != rows.stop-rows.start:
            raise ValueError("KPool selection did not preserve query rows")
        if result is None:
            result = selected.new_full((q_fp8.shape[0],selected.shape[1]),-1)
        result[rows].copy_(selected)
        # Do not retain the old allocation while the next logits call allocates.
        del logits, selected
    return result
