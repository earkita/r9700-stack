"""Standalone gfx1201 shared-page-table correctness and peak-memory probe.

Run in the pinned SGLang image. This does not measure model throughput.
"""

import argparse
import json
from pathlib import Path

import torch

from r9700_sglang.kpool import expand_mapped_groups, mask_pooled_logits
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    expand_pooled_groups_to_topk, append_kpool_tail_to_topk,
)


def old(groups, lengths, table, row_index, cols, seq, padded):
    mapped = table.index_select(0, row_index.long())
    valid = torch.arange(groups.shape[1], device=groups.device)[None, :] < lengths[:, None].clamp(max=min(cols, groups.shape[1]))
    value = expand_pooled_groups_to_topk(groups, valid, groups.shape[1] * 4, 4,
                                         page_table=mapped)
    if seq is not None:
        value = append_kpool_tail_to_topk(value, seq, lengths, 4, page_table=mapped)
    if padded != value.shape[0]:
        value = torch.nn.functional.pad(value, (0, 0, 0, padded - value.shape[0]), value=-1)
    return value


def cpu_reference(groups, lengths, table, row_index, cols, seq, padded):
    groups, lengths, table, row_index = [t.cpu() for t in (groups, lengths, table, row_index)]
    seq = None if seq is None else seq.cpu()
    topk = groups.shape[1] * 4
    expected = torch.full((padded, topk + (3 if seq is not None else 0)), -1, dtype=torch.int32)
    for row in range(groups.shape[0]):
        count = min(int(lengths[row]), cols, groups.shape[1])
        for group in range(count):
            for offset in range(4):
                token = max(0, min(int(groups[row, group]) * 4 + offset, table.shape[1] - 1))
                expected[row, group * 4 + offset] = table[int(row_index[row]), token]
        if seq is not None:
            history = min(int(lengths[row]) * 4, topk)
            for offset in range(int(seq[row]) % 4):
                token = max(0, min(int(lengths[row]) * 4 + offset, table.shape[1] - 1))
                expected[row, history + offset] = table[int(row_index[row]), token]
    return expected


def inputs(rows, budget, cols, table_cols, strided=False):
    lengths = (torch.arange(rows, device="cuda", dtype=torch.int32) * 137) % (cols + 1)
    if rows > 1:
        lengths[1] = cols
    groups = (torch.arange(budget, device="cuda", dtype=torch.int32)[None, :] * 7
              + torch.arange(rows, device="cuda", dtype=torch.int32)[:, None] * 13) % max(cols, 1)
    table = (torch.arange(table_cols * (2 if strided else 1), device="cuda", dtype=torch.int32)[None, :] * 3
             + torch.tensor([7000000, 3000000, 1000000], device="cuda", dtype=torch.int32)[:, None])
    if strided:
        table = table[:, ::2]
    index = torch.arange(rows * 2, device="cuda", dtype=torch.int32)[::2] % 3
    seq = lengths * 4 + torch.arange(rows, device="cuda", dtype=torch.int32) % 4
    return groups, lengths, table, index, seq


def old_mask(logits, lengths, starts, budget):
    cols = logits.shape[1]
    positions = torch.arange(cols, device=logits.device, dtype=torch.int64)
    starts = torch.zeros_like(lengths, dtype=torch.int64) if starts is None else starts.long()
    gather = starts[:, None] + positions[None, :]
    selected = torch.gather(logits, 1, gather.clamp(max=cols - 1))
    valid = (positions[None, :] < lengths[:, None]) & (gather < cols)
    selected = selected.masked_fill(~valid, -float("inf"))
    if cols < budget:
        return torch.nn.functional.pad(selected, (0, budget - cols), value=-float("inf"))
    return selected


def check_scores():
    cases = []
    for cols in (0, 2, 130, 1024):
        logits = torch.randn(4, cols * 2, device="cuda")[:, ::2]
        if cols:
            bits = torch.tensor([0x80000000, 0x7FC12345, 0xFF800000, 0x7F800000],
                                device="cuda", dtype=torch.int64).int().view(torch.float32)
            logits[:, 0] = bits
        lengths = torch.tensor([0, 1, cols, cols + 5], device="cuda", dtype=torch.int32)
        for starts in (None, torch.tensor([0, 1, max(cols - 2, 0), cols + 1], device="cuda", dtype=torch.int32)):
            expected = old_mask(logits, lengths, starts, 128)
            actual = mask_pooled_logits(logits, lengths, starts, min_cols=128)
            assert torch.equal(expected.view(torch.int32), actual.view(torch.int32))
            cases.append(dict(cols=cols, row_starts=starts is not None, exact_bits=True))
    logits = torch.randn(4, 1024, device="cuda")
    lengths = torch.tensor([0, 1, 1024, 512], device="cuda", dtype=torch.int32)
    starts = torch.zeros_like(lengths)
    mask_pooled_logits(logits, lengths, starts, min_cols=512)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = mask_pooled_logits(logits, lengths, starts, min_cols=512)
    for i in range(8):
        logits.add_(.125)
        starts.add_(31).remainder_(1024)
        lengths.add_(137).remainder_(1025)
        graph.replay()
        expected = old_mask(logits, lengths, starts, 512)
        assert torch.equal(output.view(torch.int32), expected.view(torch.int32))
    del graph, output, expected, logits, lengths, starts, actual
    peaks = []
    for cols in (40448, 65536):
        logits = torch.randn(2048, cols, device="cuda")
        lengths = torch.full((2048,), cols, device="cuda", dtype=torch.int32)
        starts = torch.zeros_like(lengths)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        expected = old_mask(logits, lengths, starts, 512)
        torch.cuda.synchronize()
        old_peak = torch.cuda.max_memory_allocated() - base
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        actual = mask_pooled_logits(logits, lengths, starts, min_cols=512)
        torch.cuda.synchronize()
        new_peak = torch.cuda.max_memory_allocated() - base
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        assert new_peak == actual.numel() * actual.element_size()
        peaks.append(dict(rows=2048, cols=cols, old_extra_peak_bytes=old_peak,
                          new_extra_peak_bytes=new_peak, exact_bits=True))
        del logits, lengths, starts, expected, actual
    return dict(cases=cases, graph_replays=8, large_cases=peaks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    assert torch.version.hip is not None
    torch.manual_seed(482317)
    records = []
    for rows, budget, cols in ((1, 128, 0), (4, 128, 2), (7, 128, 130), (17, 512, 1024)):
        for tail in (False, True):
            groups, lengths, table, index, seq = inputs(rows, budget, cols, max(4 * cols + 4, 7), True)
            seq = seq if tail else None
            actual = expand_mapped_groups(groups, lengths, table, index, pool_size=4,
                                         logit_cols=cols, seq_lens=seq, out_rows=rows + 2)
            expected = cpu_reference(groups, lengths, table, index, cols, seq, rows + 2)
            assert torch.equal(actual.cpu(), expected)
            assert torch.equal(actual, old(groups, lengths, table, index, cols, seq, rows + 2))
            records.append(dict(rows=rows, group_budget=budget, logit_cols=cols, tail=tail, exact=True))
    # Replay must read changed row IDs, groups and tail metadata at stable addresses.
    groups, lengths, table, index, seq = inputs(4, 512, 1024, 4100)
    expand_mapped_groups(groups, lengths, table, index, pool_size=4, logit_cols=1024, seq_lens=seq)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = expand_mapped_groups(groups, lengths, table, index, pool_size=4, logit_cols=1024, seq_lens=seq)
    for i in range(8):
        index.add_(1).remainder_(3)
        groups.add_(19).remainder_(1024)
        lengths.add_(73).remainder_(1025)
        seq.copy_(lengths * 4 + i % 4)
        graph.replay()
        assert torch.equal(output.cpu(), cpu_reference(groups, lengths, table, index, 1024, seq, 4))
    del graph, output, groups, lengths, table, index, seq
    torch.cuda.empty_cache()
    # Reproduce the exact failed allocation scale: 2048 queries × 262400 ints.
    groups, lengths, table, index, seq = inputs(2048, 512, 65536, 262400)
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    expected = old(groups, lengths, table, index, 65536, seq, 2048)
    torch.cuda.synchronize()
    old_peak = torch.cuda.max_memory_allocated() - base
    # Keep the reference output; reset the peak relative to that resident base.
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = expand_mapped_groups(groups, lengths, table, index, pool_size=4,
                                 logit_cols=65536, seq_lens=seq)
    torch.cuda.synchronize()
    new_peak = torch.cuda.max_memory_allocated() - base
    assert torch.equal(actual, expected)
    assert new_peak < 32 * 1024**2, new_peak
    score_checks = check_scores()
    result = dict(status="pass", exact_cases=records, graph_replays=8,
                  large_case=dict(rows=2048, table_cols=262400, output_cols=2051,
                                  old_extra_peak_bytes=old_peak, new_extra_peak_bytes=new_peak,
                                  exact=True), score_mask=score_checks,
                  scope="exact score/integer correctness and allocation only, not model speed")
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
