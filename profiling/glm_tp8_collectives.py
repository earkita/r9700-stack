"""Eight-rank graph-mode RCCL diagnostic for GLM-sized messages.

Run ONLY after exclusive access to all eight GPU is approved, K4 is stopped,
and the mandatory all-GPU PCIe idle/loaded evidence is arranged. This tool
does not start/stop servers and does not claim to identify trace NCCL events.

    R9K_EXCLUSIVE_GPU_TEST=1 torchrun --standalone --nproc-per-node=8 \
        profiling/glm_tp8_collectives.py --rows 1,5,20

Default all-reduce sizes are BF16 [rows, 4096] (8, 40, 160 KiB). An optional
all-gather input size must come from an instrumented production call; do not
mistake the default all-reduce shapes for observed NCCL message metadata.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist


def graph_us(call, rank, repetitions):
    for _ in range(4):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    own_us = start.elapsed_time(end) * 1000 / (20 * repetitions)
    samples = [None] * 8
    dist.all_gather_object(samples, {"rank": rank, "us": own_us})
    return samples


def main():
    if os.environ.get("R9K_EXCLUSIVE_GPU_TEST") != "1":
        raise SystemExit("exclusive eight-GPU test requires R9K_EXCLUSIVE_GPU_TEST=1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,5,20")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--gather-local-elements", type=int, default=0,
                        help="exact observed per-rank all-gather elements; 0 skips gather")
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise SystemExit(f"expected eight ranks, got {world}")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    rows = [int(value) for value in args.rows.split(",")]
    if any(value < 1 for value in rows):
        parser.error("rows must contain positive integers")
    for nrows in rows:
        shape = (nrows, args.hidden)
        x = torch.full(shape, rank + 1, dtype=torch.bfloat16, device=device)
        dist.all_reduce(x)
        torch.cuda.synchronize()
        if not bool(torch.all(x == 36)):
            raise RuntimeError(f"all-reduce incorrect on rank {rank}, shape={shape}")
        # Zero is stable under repeated in-place graph replays.
        timed = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        samples = graph_us(lambda: dist.all_reduce(timed), rank, args.repetitions)
        if rank == 0:
            print(json.dumps({"op": "all_reduce", "dtype": "bf16", "shape": shape,
                              "bytes_per_rank": timed.numel() * 2,
                              "graph_repetitions": args.repetitions,
                              "per_rank_us": samples,
                              "collective_latency_us": max(item["us"] for item in samples),
                              "correctness": "pass"}), flush=True)
    if args.gather_local_elements:
        n = args.gather_local_elements
        x = torch.full((n,), rank + 1, dtype=torch.bfloat16, device=device)
        out = torch.empty((world * n,), dtype=x.dtype, device=device)
        dist.all_gather_into_tensor(out, x)
        torch.cuda.synchronize()
        if not all(bool(torch.all(out[i*n:(i+1)*n] == i + 1)) for i in range(world)):
            raise RuntimeError(f"all-gather incorrect on rank {rank}, local_elements={n}")
        samples = graph_us(lambda: dist.all_gather_into_tensor(out, x), rank, args.repetitions)
        if rank == 0:
            print(json.dumps({"op": "all_gather", "dtype": "bf16", "local_elements": n,
                              "bytes_per_rank": n * 2, "graph_repetitions": args.repetitions,
                              "per_rank_us": samples,
                              "collective_latency_us": max(item["us"] for item in samples),
                              "correctness": "pass"}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
