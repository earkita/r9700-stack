#!/usr/bin/env python3
"""Isolated gfx1201 GLM indexer reader tile diagnostic (not model decode).

Run only with exclusive access to the selected R9700 after PCIe preflight.
Inputs match the C1/C4 DFlash K4 decode row count and 128K/256K k-pool
lengths. The existing cache round-trip test validates writer/reader arithmetic;
this script compares full logits and top-k against TILE=64 before timing
captured graph replays. It never changes the production profile.
"""

import argparse
import json
from pathlib import Path
from statistics import median


def run(args):
    import torch
    from vllm.v1.worker.workspace import init_workspace_manager
    from r9700_vllm.attn.glm_paged_logits import paged_logits

    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx1201":
        raise RuntimeError(f"expected gfx1201, got {arch}")
    init_workspace_manager(torch.cuda.current_device())
    torch.manual_seed(1234)
    page = 64
    b, next_n = args.concurrency, 5
    max_pools = args.max_context_tokens // 4
    actual_pools = args.context_tokens // 4
    if args.context_tokens % 4 or args.max_context_tokens % 4:
        raise ValueError("context token counts must be divisible by index_kpool=4")
    if not 0 < actual_pools <= max_pools:
        raise ValueError("context must fit max context")
    pages_per_seq = (max_pools + page - 1) // page
    blocks = b * pages_per_seq
    cache = torch.empty((blocks, page, 1, 132), dtype=torch.uint8, device="cuda")
    flat = cache.view(blocks, -1)
    keys = (torch.randn((blocks, page * 128), device="cuda") * 0.1).to(
        torch.float8_e4m3fn)
    flat[:, : page * 128].copy_(keys.view(torch.uint8))
    flat[:, page * 128 :].view(torch.float32).fill_(1.0)
    del keys
    q = torch.randn((b, next_n, 32, 128), device="cuda").to(torch.float8_e4m3fn)
    weights = torch.rand((b * next_n, 32), device="cuda")
    lengths = torch.full((b,), actual_pools, dtype=torch.int32, device="cuda")
    table = torch.arange(blocks, dtype=torch.int32, device="cuda").reshape(
        b, pages_per_seq)

    def call(tile):
        return paged_logits(q, cache, weights, lengths, table, None, max_pools, tile=tile)

    reference = call(64).clone()
    torch.cuda.synchronize()
    results = []
    # 32 was screened out in the first frozen pilot: it changed some FP32
    # logits by up to 1.9e-6. Keep the passing 64/128/256 set for v2.
    for tile in (64, 128, 256, 64):
        actual = call(tile).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for row in range(b * next_n):
            expected = reference[row].topk(512).indices
            observed = actual[row].topk(512).indices
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call(tile)
        for _ in range(args.warmups):
            graph.replay()
        torch.cuda.synchronize()
        times_ms = []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))
        torch.testing.assert_close(captured, reference, rtol=0, atol=0)
        results.append({"tile": tile, "times_ms": times_ms,
                        "median_ms": median(times_ms), "min_ms": min(times_ms),
                        "max_ms": max(times_ms), "exact_logits_and_topk": True})
        print(f"tile={tile}: median={median(times_ms):.4f} ms "
              f"range={min(times_ms):.4f}-{max(times_ms):.4f} ms", flush=True)
    return {"kind": "isolated_graph_replay_microbenchmark", "device": torch.cuda.get_device_name(0),
            "arch": arch, "context_tokens": args.context_tokens,
            "max_context_tokens": args.max_context_tokens, "index_kpool": 4,
            "concurrency": b, "rows": b * next_n, "page": page,
            "warmups": args.warmups, "repeats": args.repeats, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-tokens", type=int, default=130816)
    parser.add_argument("--max-context-tokens", type=int, default=262144)
    parser.add_argument("--concurrency", type=int, choices=(1, 4), default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 3:
        parser.error("need at least one warmup and three repeats")
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
