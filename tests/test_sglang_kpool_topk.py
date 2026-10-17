"""ROCm KPool top-k selection/expansion probe in the pinned SGLang image."""

import json

import torch

from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    topk_from_pooled_history_logits,
)


def main():
    assert torch.version.hip is not None
    device = "cuda"
    pool_size, group_topk = 4, 128
    logits = torch.arange(130, device=device, dtype=torch.float32)[None, :]
    lengths = torch.tensor([130], device=device, dtype=torch.int32)
    result = topk_from_pooled_history_logits(
        logits, lengths, pool_size, group_topk * pool_size
    )
    chosen = result[0].reshape(group_topk, pool_size)[:, 0].cpu().tolist()
    assert chosen == [group * pool_size for group in range(129, 1, -1)]

    short = topk_from_pooled_history_logits(
        logits, torch.tensor([2], device=device, dtype=torch.int32),
        pool_size, group_topk * pool_size,
        seq_lens=torch.tensor([11], device=device, dtype=torch.int32),
    )
    assert short[0, :11].cpu().tolist() == list(range(11))
    assert torch.all(short[0, 11:] == -1)

    page_table = (torch.arange(521, device=device, dtype=torch.int32) + 1000)[None, :]
    mapped = topk_from_pooled_history_logits(
        logits, lengths, pool_size, group_topk * pool_size,
        page_table=torch.cat((torch.zeros_like(page_table), page_table)),
        page_table_row_index=torch.tensor([1], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([521], device=device, dtype=torch.int32),
    )
    assert mapped[0, :4].cpu().tolist() == [1516, 1517, 1518, 1519]
    assert mapped[0, 512].item() == 1520

    ragged = topk_from_pooled_history_logits(
        logits, torch.tensor([129], device=device, dtype=torch.int32),
        pool_size, group_topk * pool_size,
        row_starts=torch.tensor([1], device=device, dtype=torch.int32),
        topk_offsets=torch.tensor([4000], device=device, dtype=torch.int32),
    )
    assert ragged[0, :4].cpu().tolist() == [4512, 4513, 4514, 4515]
    assert 4000 not in ragged[0].cpu().tolist()
    print(json.dumps({"topk_cases": 4, "status": "pass"}))


if __name__ == "__main__":
    main()
