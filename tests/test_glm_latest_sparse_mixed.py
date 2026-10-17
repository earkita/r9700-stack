"""The latest vLLM metadata can include dense prefill rows beside sparse decode."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from r9700_vllm.compat.gate import vllm_commit


@pytest.mark.skipif(not vllm_commit().startswith("77154ca6"),
                    reason="latest-vLLM sparse metadata contract")
def test_sparse_remap_uses_query_rows_in_mixed_batch(monkeypatch):
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as backend
    from r9700_vllm.attn import glm_sparse_fp8 as sparse_ops
    from r9700_vllm.compat.glm_sparse_fp8 import forward_fp8

    rows, batch_rows, topk = 10, 68, 128
    query = torch.zeros((rows, 8, 512), dtype=torch.bfloat16)
    cache = torch.zeros((2, 64, 512), dtype=torch.uint8)
    owner = SimpleNamespace(
        topk_indices_buffer=torch.zeros((batch_rows, topk), dtype=torch.int32),
        num_heads=8, kv_lora_rank=512, sinks=None, scale=512 ** -0.5,
        index_epoch=1,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=batch_rows, block_size=64, topk_tokens=topk,
        req_id_per_token=torch.zeros(rows, dtype=torch.int32),
        block_table=torch.zeros((4, 2), dtype=torch.int32),
        paged_kv_indptr=torch.zeros(rows + 1, dtype=torch.int32),
        paged_kv_indices=torch.zeros(rows * topk, dtype=torch.int32),
        attn_out_dtype=torch.bfloat16, remapped_buf=None, remapped_epoch=0,
    )
    seen = {}

    monkeypatch.setattr(backend, "flat_kv_row_view", lambda kv, block: (kv, block))
    monkeypatch.setattr(backend, "fit_kpool_indices_to_aiter",
                        lambda indices, width: seen.setdefault("indices", indices))

    def remap(req_ids, block_table, indices, indptr, output, **kwargs):
        seen["req_ids"] = req_ids
        seen["remapped_indices"] = indices

    monkeypatch.setattr(backend, "triton_convert_req_index_to_global_index", remap)
    monkeypatch.setattr(backend.AiterMLAHelper, "get_mla_padded_q", staticmethod(lambda heads, q: q))
    monkeypatch.setattr(backend.AiterMLAHelper, "get_mla_unpadded_o", staticmethod(lambda heads, q: q))
    monkeypatch.setattr(sparse_ops, "rocm_sparse_attn_prefill",
                        lambda **kwargs: kwargs["output"].zero_())

    result, lse = forward_fp8(owner, query, cache, metadata,
                              SimpleNamespace(_k_scale=torch.tensor(1.0)), latest=True)
    assert seen["req_ids"].shape == (rows,)
    assert seen["indices"].shape == seen["remapped_indices"].shape == (rows, topk)
    assert result.shape == (rows, 8, 512) and lse is None
